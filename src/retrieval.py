import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch

from src.garment_detector import GarmentDetector

logger = logging.getLogger(__name__)


def _generate_scene_crops(image: Image.Image) -> List[Tuple[str, Image.Image]]:
    """Generate multiple overlapping crops from a scene image.

    This multi-crop strategy bridges the domain gap between lifestyle scene images
    and isolated product catalog images by extracting focused regions that better
    match how products appear in the catalog.

    Returns:
        List of (crop_name, PIL.Image) tuples
    """
    w, h = image.size
    crops = []

    # Full image (baseline)
    crops.append(("full", image))

    # Center crop — product is often in the center
    cx, cy = w // 2, h // 2
    crop_w, crop_h = int(w * 0.6), int(h * 0.7)
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, cx + crop_w // 2)
    y2 = min(h, cy + crop_h // 2)
    crops.append(("center", image.crop((x1, y1, x2, y2))))

    # Upper body region — tops, shirts, jackets
    crops.append(("upper", image.crop((
        int(w * 0.1), int(h * 0.02),
        int(w * 0.9), int(h * 0.52)
    ))))

    # Lower body region — pants, skirts, shoes
    crops.append(("lower", image.crop((
        int(w * 0.1), int(h * 0.40),
        int(w * 0.9), int(h * 0.95)
    ))))

    # Left half — outfit element
    crops.append(("left", image.crop((
        0, int(h * 0.05),
        int(w * 0.55), int(h * 0.95)
    ))))

    # Right half — outfit element
    crops.append(("right", image.crop((
        int(w * 0.45), int(h * 0.05),
        w, int(h * 0.95)
    ))))

    # Tight center (for close-up product shots in scenes)
    crops.append(("tight_center", image.crop((
        int(w * 0.2), int(h * 0.15),
        int(w * 0.8), int(h * 0.85)
    ))))

    return crops


class SceneRetrieval:
    """Scene-based and garment-level visual product retrieval system.

    Uses multi-crop fusion to bridge the scene-to-product domain gap:
    multiple overlapping regions from the scene are encoded and their
    retrieval scores are fused (max-score per product across crops)
    for significantly better accuracy than single-shot full-scene matching.
    """

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

    def _encode_pil_batch(self, images: List[Image.Image]) -> List[Optional[np.ndarray]]:
        """Batch-encode multiple PIL images for efficiency."""
        results = []
        tensors = []
        valid_indices = []

        for i, img in enumerate(images):
            try:
                img = img.convert("RGB")
                t = self.encoder.preprocess(img)
                tensors.append(t)
                valid_indices.append(i)
            except Exception as e:
                logger.debug(f"Could not preprocess crop {i}: {e}")

        # Initialize all as None
        results = [None] * len(images)

        if not tensors:
            return results

        try:
            with torch.no_grad():
                batch = torch.stack(tensors).to(self.encoder.device)
                features = self.encoder.model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().numpy().astype(np.float32)

            for j, orig_idx in enumerate(valid_indices):
                results[orig_idx] = features_np[j]
        except Exception as e:
            logger.warning(f"Batch encoding failed: {e}")
            # Fallback to individual encoding
            for orig_idx in valid_indices:
                results[orig_idx] = self._encode_pil_image(images[orig_idx])

        return results

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

        # Retrieve extra to account for deduplication
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

    def retrieve_multi_crop(
        self,
        scene_image: Image.Image,
        top_k: int = 5,
        remove_duplicates: bool = True,
    ) -> Optional[List[Tuple[str, float]]]:
        """Multi-crop scene-level retrieval with score fusion.

        Generates multiple crops from the scene image, retrieves candidates from each,
        and fuses scores using max-score-per-product across all crops.
        This dramatically improves recall for scene-to-product matching.

        Args:
            scene_image: PIL Image of the scene
            top_k: Number of results to return
            remove_duplicates: Deduplicate returned product IDs

        Returns:
            List of (product_id, max_similarity) tuples, sorted by score
        """
        if self.catalog_index.get_size() == 0:
            return []

        crops = _generate_scene_crops(scene_image)
        crop_images = [c[1] for c in crops]

        # Batch-encode all crops efficiently
        embeddings = self._encode_pil_batch(crop_images)

        # Collect candidates from each crop and fuse by max score
        product_best_score: Dict[str, float] = {}
        candidates_per_crop = top_k * 3  # retrieve extra per crop

        for emb in embeddings:
            if emb is None:
                continue
            results = self.catalog_index.search(emb, top_k=candidates_per_crop)
            for pid, sim in results:
                if pid not in product_best_score or sim > product_best_score[pid]:
                    product_best_score[pid] = sim

        if not product_best_score:
            return []

        # Sort by best score and return top-K
        sorted_results = sorted(product_best_score.items(), key=lambda x: x[1], reverse=True)

        if remove_duplicates:
            return sorted_results[:top_k]
        else:
            return sorted_results[:top_k]

    def retrieve_from_scene(
        self,
        scene_image_path: Path,
        top_k: int = 5,
        remove_duplicates: bool = True,
    ) -> Optional[List[Tuple[str, float]]]:
        """Retrieve top-K products matching scene image using multi-crop fusion.

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

        try:
            scene_image = Image.open(scene_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot open scene image: {e}")
            return None

        return self.retrieve_multi_crop(scene_image, top_k=top_k, remove_duplicates=remove_duplicates)

    def retrieve_from_image(
        self,
        image_input,
        top_k: int = 5,
        remove_duplicates: bool = True,
    ) -> Optional[List[Tuple[str, float]]]:
        """Retrieve top-K products matching a PIL Image or numpy array.

        Uses multi-crop fusion for scene-level images.

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

        return self.retrieve_multi_crop(image, top_k=top_k, remove_duplicates=remove_duplicates)

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
                "icon": proposal.get("icon", "👕"),
                "recommendations": matches,
            })

        return garment_results

    def retrieve_dress(
        self,
        scene_image: Image.Image,
        top_k: int = 5,
        remove_duplicates: bool = True,
    ) -> Optional[Dict]:
        """Specifically detect and retrieve matching catalog products for the dress in the scene."""
        if self.catalog_index.get_size() == 0:
            return None

        dress_info = self.garment_detector.detect_dress(scene_image)
        if not dress_info:
            return None

        crop_emb = self._encode_pil_image(dress_info["crop_image"])
        if crop_emb is None:
            return None

        matches = self.retrieve_from_embedding(
            crop_emb, top_k=top_k, remove_duplicates=remove_duplicates
        )
        dress_info["recommendations"] = matches
        dress_info["icon"] = "👗"
        return dress_info

    def batch_retrieve(self, scene_paths: List[Path], top_k: int = 5) -> dict:
        """Batch retrieve results for multiple scene paths."""
        results = {}
        for p in scene_paths:
            res = self.retrieve_from_scene(p, top_k=top_k)
            results[str(p)] = res if res is not None else []
        return results
