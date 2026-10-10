import logging
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

COLOR_PALETTES = {
    "Black": (30, 30, 30),
    "White": (235, 235, 235),
    "Gray": (128, 128, 128),
    "Navy / Dark Blue": (25, 45, 90),
    "Light Blue": (120, 175, 220),
    "Beige / Cream": (215, 200, 175),
    "Brown / Tan": (130, 85, 55),
    "Red": (190, 40, 40),
    "Green / Olive": (70, 110, 60),
    "Yellow / Gold": (220, 190, 50),
    "Pink": (225, 140, 165),
}

PATTERN_PROMPTS = [
    ("Solid Color", "a photo of solid plain single color fabric"),
    ("Striped", "a photo of striped fabric pattern"),
    ("Floral", "a photo of floral printed fabric"),
    ("Plaid / Checkered", "a photo of plaid or checkered pattern fabric"),
    ("Graphic / Printed", "a photo of graphic printed or textured fabric"),
]


class AttributeExtractor:
    """Extracts visual attributes directly from image pixels and CLIP representations."""

    def __init__(self, clip_encoder=None):
        self.clip_encoder = clip_encoder
        self.device = getattr(clip_encoder, "device", torch.device("cpu"))
        self.pattern_features = None

        if self.clip_encoder is not None:
            self._init_patterns()

    def _init_patterns(self):
        try:
            import open_clip
            prompts = [p[1] for p in PATTERN_PROMPTS]
            tokenizer = open_clip.get_tokenizer(self.clip_encoder.model_name)
            tokens = tokenizer(prompts).to(self.device)
            with torch.no_grad():
                features = self.clip_encoder.model.encode_text(tokens)
                self.pattern_features = features / features.norm(dim=-1, keepdim=True)
        except Exception as e:
            logger.debug(f"Pattern feature initialization failed: {e}")

    def extract_dominant_color(self, image: Image.Image) -> str:
        """Extract dominant color from image pixels."""
        try:
            # Resize image for quick RGB analysis
            thumb = image.convert("RGB").resize((50, 50))
            arr = np.array(thumb).reshape(-1, 3)

            # Ignore extreme white/bright borders typical of catalog backdrops
            mask = ~((arr[:, 0] > 245) & (arr[:, 1] > 245) & (arr[:, 2] > 245))
            if mask.sum() > 50:
                fg_pixels = arr[mask]
            else:
                fg_pixels = arr

            avg_rgb = fg_pixels.mean(axis=0)

            # Find nearest palette color via Euclidean distance
            best_color = "Neutral / Multi-color"
            min_dist = float("inf")
            for name, rgb in COLOR_PALETTES.items():
                dist = np.linalg.norm(avg_rgb - np.array(rgb))
                if dist < min_dist:
                    min_dist = dist
                    best_color = name

            return best_color
        except Exception:
            return "Multi-color"

    def extract_pattern(self, image: Image.Image) -> str:
        """Extract pattern using zero-shot CLIP classification."""
        if self.clip_encoder is None or self.pattern_features is None:
            return "Solid / Subtle"

        try:
            with torch.no_grad():
                tensor = self.clip_encoder.preprocess(image).unsqueeze(0).to(self.device)
                emb = self.clip_encoder.model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                sims = (emb @ self.pattern_features.T)[0]
                best_idx = int(torch.argmax(sims))
                return PATTERN_PROMPTS[best_idx][0]
        except Exception:
            return "Solid / Subtle"

    def extract_attributes(self, image_path_or_img) -> Dict[str, str]:
        """Extract dominant color and pattern."""
        if isinstance(image_path_or_img, (str, Path)):
            try:
                img = Image.open(image_path_or_img).convert("RGB")
            except Exception:
                return {"dominant_color": "Unknown", "pattern": "Unknown"}
        else:
            img = image_path_or_img.convert("RGB")

        color = self.extract_dominant_color(img)
        pattern = self.extract_pattern(img)

        return {
            "dominant_color": color,
            "pattern": pattern,
        }


class RecommendationExplainer:
    """Generates transparent, evidence-grounded explanations for recommendations."""

    def __init__(self, catalog_index, clip_encoder=None):
        self.catalog_index = catalog_index
        self.extractor = AttributeExtractor(clip_encoder)

    def explain_recommendation(
        self,
        product_id: str,
        similarity_score: float,
        rank: int = 1,
        query_image: Optional[Image.Image] = None,
        is_exact_catalog_match: bool = False,
    ) -> dict:
        """Generate an evidence-grounded explanation for a recommendation.

        Args:
            product_id: Catalog Product ID
            similarity_score: Cosine similarity score [-1.0, 1.0]
            rank: Ranking position
            query_image: Optional scene or garment crop image
            is_exact_catalog_match: True if verified as exact identity

        Returns:
            Dict containing transparent explanation attributes and summary
        """
        metadata = self.catalog_index.get_metadata(product_id) or {}
        image_path = metadata.get("image_path")

        # Extract catalog product visual attributes
        prod_attrs = {"dominant_color": "Neutral", "pattern": "Solid"}
        if image_path and Path(image_path).exists():
            prod_attrs = self.extractor.extract_attributes(image_path)

        # Extract query visual attributes if query image provided
        query_attrs = None
        if query_image is not None:
            query_attrs = self.extractor.extract_attributes(query_image)

        # Construct evidence-based reasons
        evidence_points = []
        if query_attrs:
            if query_attrs["dominant_color"] == prod_attrs["dominant_color"]:
                evidence_points.append(f"Matching color palette ({prod_attrs['dominant_color']})")
            else:
                evidence_points.append(
                    f"Harmonious tone (Query: {query_attrs['dominant_color']} vs Catalog: {prod_attrs['dominant_color']})"
                )

            if query_attrs["pattern"] == prod_attrs["pattern"]:
                evidence_points.append(f"Shared texture/pattern ({prod_attrs['pattern']})")
        else:
            evidence_points.append(f"Dominant color: {prod_attrs['dominant_color']}")
            evidence_points.append(f"Pattern style: {prod_attrs['pattern']}")

        # Similarity interpretation
        if similarity_score >= 0.82:
            tier = "High visual correspondence"
        elif similarity_score >= 0.70:
            tier = "Substantial stylistic similarity"
        elif similarity_score >= 0.55:
            tier = "Moderate visual similarity"
        else:
            tier = "Broad thematic / silhouette affinity"

        match_label = (
            "Verified Exact Product Match"
            if is_exact_catalog_match
            else "Visually Similar Recommendation (Not Verified Exact Identity)"
        )

        return {
            "product_id": product_id,
            "rank": rank,
            "cosine_similarity": round(float(similarity_score), 4),
            "similarity_tier": tier,
            "match_type": match_label,
            "is_exact_match": is_exact_catalog_match,
            "dominant_color": prod_attrs["dominant_color"],
            "pattern": prod_attrs["pattern"],
            "evidence": evidence_points,
            "summary": (
                f"{tier} (Cosine similarity: {similarity_score:.4f}). "
                f"Visual evidence: {'; '.join(evidence_points)}. "
                f"Status: {match_label}."
            ),
            "image_path": image_path,
        }

    def explain_batch(
        self,
        results: List[Tuple[str, float]],
        query_image: Optional[Image.Image] = None,
        ground_truth_ids: Optional[set] = None,
    ) -> List[dict]:
        """Generate explanations for a batch of retrieved results."""
        explanations = []
        gt = ground_truth_ids or set()
        for rank, (product_id, sim) in enumerate(results, 1):
            is_exact = product_id in gt
            exp = self.explain_recommendation(
                product_id=product_id,
                similarity_score=sim,
                rank=rank,
                query_image=query_image,
                is_exact_catalog_match=is_exact,
            )
            explanations.append(exp)
        return explanations
