import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch

from src.garment_detector import GarmentDetector

logger = logging.getLogger(__name__)

# Category compatibility matrix: maps query category to compatible categories
CATEGORY_COMPATIBILITY = {
    "Jacket / Outerwear": {
        "preferred": ["Jacket / Outerwear"],
        "compatible": [],
        "forbidden": ["Footwear", "Skirt", "Dress", "Bag / Handbag", "Pants / Trousers"],
    },
    "Top / T-shirt": {
        "preferred": ["Top / T-shirt"],
        "compatible": [],
        "forbidden": ["Jacket / Outerwear", "Footwear", "Pants / Trousers", "Skirt", "Bag / Handbag", "Dress"],
    },
    "Pants / Trousers": {
        "preferred": ["Pants / Trousers"],
        "compatible": [],
        "forbidden": ["Skirt", "Jacket / Outerwear", "Top / T-shirt", "Footwear", "Bag / Handbag", "Dress"],
    },
    "Dress": {
        "preferred": ["Dress"],
        "compatible": ["Skirt"],
        "forbidden": ["Pants / Trousers", "Footwear", "Bag / Handbag", "Jacket / Outerwear"],
    },
    "Skirt": {
        "preferred": ["Skirt"],
        "compatible": ["Dress"],
        "forbidden": ["Pants / Trousers", "Jacket / Outerwear", "Top / T-shirt", "Footwear", "Bag / Handbag"],
    },
    "Footwear": {
        "preferred": ["Footwear"],
        "compatible": [],
        "forbidden": ["Jacket / Outerwear", "Top / T-shirt", "Pants / Trousers", "Dress", "Skirt", "Bag / Handbag"],
    },
    "Bag / Handbag": {
        "preferred": ["Bag / Handbag"],
        "compatible": [],
        "forbidden": ["Footwear", "Pants / Trousers", "Jacket / Outerwear", "Dress", "Skirt", "Top / T-shirt"],
    },
}

COLOR_HARMONY = {
    "Black": {"compatible": ["Black", "Dark Grey", "Grey"], "clashing": ["Pink", "Maroon / Burgundy", "Red", "Yellow / Gold", "Orange"]},
    "Dark Grey": {"compatible": ["Dark Grey", "Black", "Grey"], "clashing": ["Pink", "Red", "Yellow / Gold", "Orange"]},
    "Grey": {"compatible": ["Grey", "Light Grey", "Dark Grey", "Black"], "clashing": ["Pink", "Red", "Maroon / Burgundy", "Orange", "Brown / Tan"]},
    "Light Grey": {"compatible": ["Light Grey", "Grey", "White"], "clashing": ["Brown / Tan", "Maroon / Burgundy", "Red", "Pink", "Orange"]},
    "White": {"compatible": ["White", "Light Grey", "Beige / Khaki"], "clashing": []},
    "Navy / Dark Blue": {"compatible": ["Navy / Dark Blue", "Blue", "Black", "Dark Grey"], "clashing": ["Yellow / Gold", "Red", "Pink", "Orange"]},
    "Blue": {"compatible": ["Blue", "Navy / Dark Blue", "Light Blue"], "clashing": ["Red", "Orange", "Pink", "Maroon / Burgundy"]},
    "Light Blue": {"compatible": ["Light Blue", "Blue", "White"], "clashing": ["Maroon / Burgundy", "Brown / Tan"]},
    "Red": {"compatible": ["Red", "Maroon / Burgundy"], "clashing": ["Green", "Olive / Dark Green", "Blue", "Pink"]},
    "Maroon / Burgundy": {"compatible": ["Maroon / Burgundy", "Red", "Brown / Tan"], "clashing": ["Light Grey", "Grey", "Blue", "Green", "Pink"]},
    "Brown / Tan": {"compatible": ["Brown / Tan", "Beige / Khaki", "Black"], "clashing": ["Pink", "Light Blue", "Purple"]},
    "Beige / Khaki": {"compatible": ["Beige / Khaki", "White", "Brown / Tan"], "clashing": ["Pink", "Purple"]},
    "Green": {"compatible": ["Green", "Olive / Dark Green", "Dark Grey"], "clashing": ["Red", "Pink", "Maroon / Burgundy"]},
    "Olive / Dark Green": {"compatible": ["Olive / Dark Green", "Green", "Black", "Dark Grey"], "clashing": ["Red", "Pink", "Maroon / Burgundy"]},
    "Pink": {"compatible": ["Pink", "White"], "clashing": ["Black", "Dark Grey", "Olive / Dark Green", "Brown / Tan"]},
    "Purple": {"compatible": ["Purple", "Dark Grey", "Black"], "clashing": ["Yellow / Gold", "Orange", "Green"]},
    "Yellow / Gold": {"compatible": ["Yellow / Gold", "White"], "clashing": ["Navy / Dark Blue", "Purple", "Black"]},
    "Orange": {"compatible": ["Orange", "Brown / Tan"], "clashing": ["Navy / Dark Blue", "Blue", "Purple"]},
}


def _generate_scene_crops(image: Image.Image) -> List[Tuple[str, Image.Image]]:
    """Generate multi-crop views of a scene for full-scene matching."""
    w, h = image.size
    crops = [("full", image)]

    # Center crop
    cx, cy = w // 2, h // 2
    crop_w, crop_h = int(w * 0.65), int(h * 0.75)
    x1, y1 = max(0, cx - crop_w // 2), max(0, cy - crop_h // 2)
    x2, y2 = min(w, cx + crop_w // 2), min(h, cy + crop_h // 2)
    crops.append(("center", image.crop((x1, y1, x2, y2))))

    # Upper body
    crops.append(("upper", image.crop((int(w * 0.08), int(h * 0.05), int(w * 0.92), int(h * 0.58)))))

    # Lower body
    crops.append(("lower", image.crop((int(w * 0.10), int(h * 0.45), int(w * 0.90), int(h * 0.95)))))

    return crops


class SceneRetrieval:
    """Category-aware fashion visual discovery and garment retrieval engine."""

    def __init__(self, encoder, catalog_index, garment_detector: Optional[GarmentDetector] = None):
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
        """Batch-encode multiple PIL images efficiently."""
        results = [None] * len(images)
        tensors = []
        valid_indices = []

        for i, img in enumerate(images):
            try:
                t = self.encoder.preprocess(img.convert("RGB"))
                tensors.append(t)
                valid_indices.append(i)
            except Exception as e:
                logger.debug(f"Preprocess failed for crop {i}: {e}")

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
            logger.warning(f"Batch encoding failed, using fallback: {e}")
            for orig_idx in valid_indices:
                results[orig_idx] = self._encode_pil_image(images[orig_idx])

        return results

    def rerank_candidates(
        self,
        raw_candidates: List[Tuple[str, float]],
        query_category: Optional[str] = None,
        query_color: Optional[str] = None,
        query_pattern: Optional[str] = None,
        query_attributes: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> List[Tuple[str, float, float, Dict[str, Any]]]:
        """Category-aware, pattern-calibrated, and style-grounded candidate reranking.

        Evaluates candidate products using:
        1. Visual embedding cosine similarity (base visual signal).
        2. Garment category compatibility (forbidding cross-category bleed like jackets for tops or pants for skirts).
        3. Color harmony / compatibility (rewarding matching/secondary tones, penalizing clashing colors).
        4. Pattern compatibility (prioritizing striped/patterned items, penalizing solid mismatches).
        5. Silhouette and sleeve compatibility (rejecting black sleeveless tops or denim jackets when querying striped/gathered tops).

        Returns:
            List of (product_id, raw_cosine_sim, rerank_score, diagnostic_dict)
        """
        scored_candidates = []
        rules = CATEGORY_COMPATIBILITY.get(query_category, {})
        preferred_cats = rules.get("preferred", [])
        compatible_cats = rules.get("compatible", [])
        forbidden_cats = rules.get("forbidden", [])

        primary_col = query_color
        secondary_col = None
        if query_attributes:
            primary_col = query_attributes.get("primary_color", query_color)
            secondary_col = query_attributes.get("secondary_color")
            query_pattern = query_pattern or query_attributes.get("pattern")

        color_rules = COLOR_HARMONY.get(primary_col, {})
        compat_colors = color_rules.get("compatible", [primary_col] if primary_col else [])
        clash_colors = color_rules.get("clashing", [])

        seen_pids = set()
        for pid, raw_sim in raw_candidates:
            if pid in seen_pids:
                continue
            seen_pids.add(pid)

            meta = self.catalog_index.get_metadata(pid) or {}
            cand_cat = meta.get("category", "")
            cand_color = meta.get("dominant_color", "")
            cand_pattern = meta.get("pattern", "Solid Color")

            # 1. Category score adjustment
            cat_delta = 0.0
            cat_status = "Neutral"
            if query_category:
                if cand_cat in preferred_cats:
                    cat_delta = 0.12
                    cat_status = "Exact Category Match"
                elif cand_cat in compatible_cats:
                    cat_delta = 0.02
                    cat_status = "Compatible Category"
                elif cand_cat in forbidden_cats:
                    cat_delta = -0.35  # Strong penalty against forbidden category
                    cat_status = "Category Mismatch"
                else:
                    cat_delta = -0.08
                    cat_status = "Unverified Category"

            # 2. Color score adjustment
            color_delta = 0.0
            color_status = "Neutral"
            if primary_col and cand_color:
                if cand_color == primary_col:
                    color_delta = 0.10
                    color_status = "Exact Color Match"
                elif secondary_col and cand_color == secondary_col:
                    color_delta = 0.06
                    color_status = f"Secondary Tone Match ({secondary_col})"
                elif cand_color in compat_colors:
                    color_delta = 0.04
                    color_status = "Harmonious Color"
                elif cand_color in clash_colors:
                    color_delta = -0.20  # Penalty for clashing color
                    color_status = "Color Clash"

            # 3. Pattern score adjustment
            pattern_delta = 0.0
            pattern_status = "Neutral"
            if query_pattern:
                if cand_pattern == query_pattern:
                    pattern_delta = 0.10
                    pattern_status = f"Pattern Match ({cand_pattern})"
                elif query_pattern == "Striped" and cand_pattern == "Solid Color":
                    pattern_delta = -0.15
                    pattern_status = "Pattern Mismatch (Solid vs Striped)"
                elif query_pattern == "Striped" and cand_pattern != "Striped":
                    pattern_delta = -0.08
                    pattern_status = f"Pattern Mismatch ({cand_pattern} vs Striped)"
                elif query_pattern == "Solid Color" and cand_pattern != "Solid Color":
                    pattern_delta = -0.08
                    pattern_status = f"Pattern Mismatch ({cand_pattern} vs Solid)"

            # 4. Style, Silhouette & Sleeve adjustment
            style_delta = 0.0
            style_status = "Neutral"
            if query_attributes:
                q_sil = query_attributes.get("silhouette")
                q_sleeve = query_attributes.get("sleeve")

                # If query is gathered/peplum top:
                if q_sil == "Gathered / Peplum / Ruffled":
                    if cand_cat == "Jacket / Outerwear":
                        style_delta -= 0.15
                        style_status = "Denim/Outerwear Mismatch for Gathered Top"
                    elif cand_cat == "Top / T-shirt" and cand_pattern in ["Striped", "Graphic / Printed"]:
                        style_delta += 0.06
                        style_status = "Gathered / Peplum Silhouette Compatible"

                # If query has sleeves: penalize plain black sleeveless tops
                if q_sleeve in ["Short Sleeve", "Long Sleeve"]:
                    if cand_color == "Black" and cand_cat == "Top / T-shirt" and cand_pattern == "Solid Color":
                        style_delta -= 0.15
                        style_status = "Sleeve & Tone Mismatch (Black Sleeveless vs Sleeved Striped)"

            # Calibrated combined score
            rerank_score = float(raw_sim + cat_delta + color_delta + pattern_delta + style_delta)

            diag = {
                "product_id": pid,
                "raw_sim": round(float(raw_sim), 4),
                "rerank_score": round(rerank_score, 4),
                "cand_category": cand_cat,
                "cand_color": cand_color,
                "cand_pattern": cand_pattern,
                "category_status": cat_status,
                "color_status": color_status,
                "pattern_status": pattern_status,
                "style_status": style_status,
                "cat_delta": cat_delta,
                "color_delta": color_delta,
                "pattern_delta": pattern_delta,
                "style_delta": style_delta,
            }

            scored_candidates.append((pid, float(raw_sim), rerank_score, diag))

        # Sort strictly by rerank_score descending
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        return scored_candidates[:top_k]

    def retrieve_garments(
        self,
        scene_image: Image.Image,
        top_k_per_garment: int = 3,
        remove_duplicates: bool = True,
    ) -> List[Dict]:
        """Garment-level visual product retrieval with category and color reranking.

        Extracts anatomically localized garment crops and retrieves category-appropriate
        catalog recommendations for each detected item.
        """
        if self.catalog_index.get_size() == 0:
            return []

        garment_proposals = self.garment_detector.extract_garments(scene_image)
        garment_results = []

        for proposal in garment_proposals:
            crop_img = proposal["image"]
            query_cat = proposal["label"]
            query_col = proposal.get("dominant_color", "")

            # Extract fine-grained attributes using attribute_extractor if available
            query_attrs = proposal.get("attributes")
            if not query_attrs and hasattr(self.garment_detector, "attribute_extractor") and self.garment_detector.attribute_extractor is not None:
                query_attrs = self.garment_detector.attribute_extractor.extract_attributes(crop_img)
            query_pat = proposal.get("pattern") or (query_attrs.get("pattern") if query_attrs else "")

            crop_emb = self._encode_pil_image(crop_img)
            if crop_emb is None:
                continue

            # Retrieve candidate pool using category-aware search from catalog_index
            candidate_pool_size = max(top_k_per_garment * 10, 80)
            if hasattr(self.catalog_index, "search_candidates"):
                raw_candidates = self.catalog_index.search_candidates(
                    crop_emb, target_category=query_cat, candidate_pool_size=candidate_pool_size
                )
            else:
                raw_candidates = self.catalog_index.search(crop_emb, top_k=candidate_pool_size)

            # Deduplicate raw candidates
            seen = set()
            deduped_raw = []
            for pid, sim in raw_candidates:
                if pid not in seen:
                    seen.add(pid)
                    deduped_raw.append((pid, sim))

            # Apply Category, Color, Pattern, and Style Reranking
            reranked = self.rerank_candidates(
                raw_candidates=deduped_raw,
                query_category=query_cat,
                query_color=query_col,
                query_pattern=query_pat,
                query_attributes=query_attrs,
                top_k=top_k_per_garment,
            )

            # Format final recommendations: (pid, raw_sim, rerank_score) tuples with attached rerank metadata
            recommendations = [(item[0], item[1], item[2]) for item in reranked]
            diagnostics = [item[3] for item in reranked]

            garment_results.append({
                "garment_label": query_cat,
                "region": proposal["region"],
                "box": proposal["box"],
                "crop_image": crop_img,
                "confidence": proposal["confidence"],
                "dominant_color": query_col,
                "pattern": query_pat,
                "attributes": query_attrs or {},
                "icon": proposal.get("icon", "🏷️"),
                "is_uncertain": proposal.get("is_uncertain", False),
                "recommendations": recommendations,
                "diagnostics": diagnostics,
            })

        return garment_results

    def retrieve_from_image(
        self,
        image_input,
        top_k: int = 3,
        remove_duplicates: bool = True,
    ) -> Optional[List[Tuple[str, float]]]:
        """Full-scene baseline visual retrieval using multi-crop score fusion."""
        if isinstance(image_input, np.ndarray):
            try:
                if image_input.ndim == 3 and image_input.shape[-1] == 4:
                    image_input = image_input[:, :, :3]
                image = Image.fromarray(image_input.astype(np.uint8), "RGB")
            except Exception:
                return None
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            return None

        crops = _generate_scene_crops(image)
        crop_images = [c[1] for c in crops]
        embeddings = self._encode_pil_batch(crop_images)

        product_best_score: Dict[str, float] = {}
        for emb in embeddings:
            if emb is None:
                continue
            results = self.catalog_index.search(emb, top_k=top_k * 4)
            for pid, sim in results:
                if pid not in product_best_score or sim > product_best_score[pid]:
                    product_best_score[pid] = sim

        if not product_best_score:
            return []

        sorted_results = sorted(product_best_score.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
