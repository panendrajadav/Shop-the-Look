import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch

from src.garment_detector import GarmentDetector

logger = logging.getLogger(__name__)


class SceneRetrieval:
    """Scene-based and garment-level visual product retrieval system."""

    def __init__(self, encoder, catalog_index, garment_detector: Optional[GarmentDetector] = None):
        """Initialize retrieval system.

        Args:
            encoder: CLIPImageEncoder instance
            catalog_index: ProductIndex instance
            garment_detector: Optional GarmentDetector instance
        """
        self.encoder = encoder
        self.catalog_index = catalog_index
        self.garment_detector = garment_detector or GarmentDetector(clip_encoder=encoder)

    def _encode_pil_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """Encode a PIL image into a normalized 1D embedding vector."""
        try:
            image = image.convert("RGB")
            with torch.no_grad():
                tensor = self.encoder.preprocess(image).unsqueeze(0).to(self.encoder.device)
                features = self.encoder.model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()[0].astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to encode PIL image: {e}")
            return None

    def retrieve_from_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        remove_duplicates: bool = True,
    ) -> List[Tuple[str, float]]:
        """Retrieve top-K products for a precomputed query embedding."""
        if self.catalog_index.get_size() == 0:
            logger.warning("Catalog index is empty")
            return []

        # Retrieve a few extra to account for deduplication
        raw_results = self.catalog_index.search(query_embedding, top_k=top_k * 3)

        if not remove_duplicates:
            return raw_results[:top_k]

        seen = set()
        deduped = []
        for pid, sim in raw_results:
            if pid not in seen:
                seen.add(pid)
                deduped.append((pid, sim))
                if len(deduped) >= top_k:
                    break

        return deduped

    def retrieve_from_scene(
        self,
        scene_image_path: Path,
        top_k: int = 5,
        remove_duplicates: bool = True,
    ) -> Optional[List[Tuple[str, float]]]:
        """Retrieve top-K products matching full scene image.

        Args:
            scene_image_path: Path to scene image
            top_k: Number of results to return
            remove_duplicates: Deduplicate returned product IDs

        Returns:
            List of (product_id, cosine_similarity) tuples
        """
        scene_path = Path(scene_image_path)
        if not scene_path.exists():
            logger.warning(f"Scene image path does not exist: {scene_path}")
            return None

        emb = self.encoder.encode_image(scene_path)
        if emb is None:
            return None

        return self.retrieve_from_embedding(emb, top_k=top_k, remove_duplicates=remove_duplicates)

    def retrieve_from_image(
        self,
        image_input,
        top_k: int = 5,
        remove_duplicates: bool = True,
    ) -> Optional[List[Tuple[str, float]]]:
        """Retrieve top-K products matching a PIL Image or numpy array.

        Args:
            image_input: PIL.Image or numpy.ndarray (RGB)
            top_k: Number of results to return
            remove_duplicates: Deduplicate returned product IDs

        Returns:
            List of (product_id, cosine_similarity) tuples
        """
        if isinstance(image_input, np.ndarray):
            try:
                if image_input.ndim == 3 and image_input.shape[-1] == 4:
                    image_input = image_input[:, :, :3]
                image = Image.fromarray(image_input.astype(np.uint8), "RGB")
            except Exception as e:
                logger.warning(f"Could not convert numpy array to PIL Image: {e}")
                return None
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            logger.warning(f"Unsupported image input type: {type(image_input)}")
            return None

        emb = self._encode_pil_image(image)
        if emb is None:
            return None

        return self.retrieve_from_embedding(emb, top_k=top_k, remove_duplicates=remove_duplicates)

    def retrieve_garments(
        self,
        scene_image: Image.Image,
        top_k_per_garment: int = 3,
        remove_duplicates: bool = True,
    ) -> List[Dict]:
        """Garment-level visual product retrieval.

        Detects garment regions (tops, pants, dresses, footwear, etc.)
        and retrieves catalog matches for each detected item.

        Returns list of dicts:
        [{
            "garment_label": "Top / Shirt",
            "region": "Upper Body",
            "box": (x1, y1, x2, y2),
            "crop_image": PIL.Image,
            "confidence": float,
            "recommendations": [(product_id, similarity), ...]
        }, ...]
        """
        if self.catalog_index.get_size() == 0:
            return []

        garment_proposals = self.garment_detector.extract_garments(scene_image)
        garment_results = []

        for proposal in garment_proposals:
            crop_img = proposal["image"]
            crop_emb = self._encode_pil_image(crop_img)
            if crop_emb is None:
                continue

            matches = self.retrieve_from_embedding(
                crop_emb, top_k=top_k_per_garment, remove_duplicates=remove_duplicates
            )

            garment_results.append({
                "garment_label": proposal["label"],
                "region": proposal["region"],
                "box": proposal["box"],
                "crop_image": crop_img,
                "confidence": proposal["confidence"],
                "recommendations": matches,
            })

        return garment_results

    def batch_retrieve(self, scene_paths: List[Path], top_k: int = 5) -> dict:
        """Batch retrieve results for multiple scene paths."""
        results = {}
        for p in scene_paths:
            res = self.retrieve_from_scene(p, top_k=top_k)
            results[str(p)] = res if res is not None else []
        return results
