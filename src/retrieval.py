import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch

from src.garment_detector import GarmentDetector
from src.color_features import color_histogram

logger = logging.getLogger(__name__)

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
    "Accessory": {
        "preferred": ["Accessory"],
        "compatible": [],
        "forbidden": ["Footwear", "Pants / Trousers", "Jacket / Outerwear", "Dress", "Skirt", "Top / T-shirt", "Bag / Handbag"],
    },
    "Full Outfit / Look": {
        "preferred": ["Dress", "Jacket / Outerwear", "Top / T-shirt", "Pants / Trousers", "Skirt"],
        "compatible": ["Skirt", "Footwear", "Bag / Handbag"],
        "forbidden": [],
    },
}

CANDIDATE_POOL_SIZE = 200

COLOR_WEIGHT = 0.05
PATTERN_WEIGHT = 0.02
ITEM_WEIGHT = 0.03
CATEGORY_MISMATCH_PENALTY = -0.35


def _generate_scene_crops(image: Image.Image) -> List[Tuple[str, Image.Image]]:
    """Generate multi-crop views of a scene for full-scene matching."""
    w, h = image.size
    crops = [("full", image)]

    cx, cy = w // 2, h // 2
    crop_w, crop_h = int(w * 0.65), int(h * 0.75)
    x1, y1 = max(0, cx - crop_w // 2), max(0, cy - crop_h // 2)
    x2, y2 = min(w, cx + crop_w // 2), min(h, cy + crop_h // 2)
    crops.append(("center", image.crop((x1, y1, x2, y2))))

    crops.append(("upper", image.crop((int(w * 0.08), int(h * 0.05), int(w * 0.92), int(h * 0.58)))))

    crops.append(("lower", image.crop((int(w * 0.10), int(h * 0.42), int(w * 0.90), int(h * 0.92)))))

    crops.append(("footwear", image.crop((int(w * 0.12), int(h * 0.78), int(w * 0.88), int(h * 0.99)))))

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

    def _encode_garment(self, crop: Image.Image) -> Optional[np.ndarray]:
        """Garment-focused query embedding.

        Averages the crop, its mirror image and its central 80% (which drops most of the
        background that body-proportion boxes pick up at their edges), then re-normalizes.
        """
        cw, ch = crop.size
        views = [crop, crop.transpose(Image.FLIP_LEFT_RIGHT)]
        if cw >= 60 and ch >= 60:
            views.append(crop.crop((int(0.10 * cw), int(0.10 * ch), int(0.90 * cw), int(0.90 * ch))))
        embs = [e for e in self._encode_pil_batch(views) if e is not None]
        if not embs:
            return None
        mean = np.mean(embs, axis=0)
        return (mean / (np.linalg.norm(mean) + 1e-8)).astype(np.float32)

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
        raw_candidates: List[Tuple],
        query_category: Optional[str] = None,
        query_color: Optional[str] = None,
        query_pattern: Optional[str] = None,
        query_attributes: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        query_item: Optional[str] = None,
        query_hist: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float, float, Dict[str, Any]]]:
        """Order the category-restricted candidates of one detected item.

        Score = ranking score from the index (domain-centred cosine similarity)
                + small adjustments for color overlap, pattern and same specific item.
        Category is enforced by the candidate pool itself; a candidate from outside the
        category (admitted only as a near-duplicate image) is penalised unless it really is a
        near-duplicate. The adjustment weights are deliberately small: on the validation
        scenes larger weights moved more labelled products down than up.

        Candidates are (product_id, cosine_similarity, ranking_score); a 2-tuple is accepted
        and its similarity used as the ranking score.

        Returns:
            List of (product_id, cosine_similarity, rerank_score, diagnostic_dict)
        """
        rules = CATEGORY_COMPATIBILITY.get(query_category, {})
        preferred_cats = rules.get("preferred", [])

        scored_candidates = []
        seen_pids = set()
        for cand in raw_candidates:
            pid, raw_sim = cand[0], float(cand[1])
            base = float(cand[2]) if len(cand) > 2 else raw_sim
            if pid in seen_pids:
                continue
            seen_pids.add(pid)

            meta = self.catalog_index.get_metadata(pid) or {}
            cand_cat = meta.get("category", "")
            cand_pattern = meta.get("pattern", "Solid Color")

            cat_delta, cat_status = 0.0, "Neutral"
            if query_category and preferred_cats:
                if cand_cat in preferred_cats:
                    cat_status = "Category Match"
                elif meta.get("category_alt") in preferred_cats:
                    cat_status = "Category Match (second reading of an ambiguous photo)"
                else:
                    cat_delta, cat_status = CATEGORY_MISMATCH_PENALTY, "Category Mismatch"

            color_delta, color_status = 0.0, "Neutral"
            color_sim = self.catalog_index.color_similarity(pid, query_hist) if hasattr(self.catalog_index, "color_similarity") else None
            if color_sim is not None:
                color_delta = COLOR_WEIGHT * (color_sim - 0.5)
                color_status = f"Color overlap {color_sim:.2f}"

            pattern_delta, pattern_status = 0.0, "Neutral"
            if query_pattern and cand_pattern == query_pattern:
                pattern_delta, pattern_status = PATTERN_WEIGHT, f"Pattern Match ({cand_pattern})"

            item_delta = ITEM_WEIGHT if (query_item and meta.get("item_name") == query_item) else 0.0

            rerank_score = base + cat_delta + color_delta + pattern_delta + item_delta
            if raw_sim >= 0.97:
                rerank_score = raw_sim + 1.0

            scored_candidates.append((pid, raw_sim, float(rerank_score), {
                "product_id": pid,
                "cosine_similarity": round(raw_sim, 4),
                "ranking_score": round(base, 4),
                "rerank_score": round(float(rerank_score), 4),
                "cand_item": meta.get("item_name", ""),
                "cand_category": cand_cat,
                "cand_color": meta.get("dominant_color", ""),
                "cand_pattern": cand_pattern,
                "category_status": cat_status,
                "color_status": color_status,
                "pattern_status": pattern_status,
                "item_match": bool(item_delta),
            }))

        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        return scored_candidates[:top_k]

    def build_queries(self, scene_image: Image.Image) -> Dict[str, Any]:
        """Detect items in a photo and turn each into a query (embedding + attributes).

        This is the expensive, label-free half of the pipeline; its output can be cached and
        matched against the catalog repeatedly.
        """
        proposals = self.garment_detector.extract_garments(scene_image)
        queries, seen = [], set()
        for p in proposals:
            emb = self._encode_garment(p["image"])
            if emb is None:
                continue
            item_name = p.get("item_name") or self.garment_detector.name_item(emb, p["label"])
            if item_name in seen:
                continue
            seen.add(item_name)
            queries.append({
                "label": p["label"],
                "item_name": item_name,
                "region": p["region"],
                "box": tuple(p["box"]),
                "crop_image": p["image"],
                "confidence": float(p["confidence"]),
                "dominant_color": p.get("dominant_color", ""),
                "pattern": self.garment_detector.name_pattern(emb),
                "icon": p.get("icon", "🏷️"),
                "is_uncertain": p.get("is_uncertain", False),
                "embedding": emb,
                "color_hist": color_histogram(p["image"], center_only=True),
            })
        w, h = scene_image.size
        covers_frame = any(q["box"] == (0, 0, w, h) for q in queries)
        return {
            "queries": queries,
            "size": (w, h),
            "full_embedding": None if covers_frame else self._encode_pil_image(scene_image),
        }

    def match_queries(
        self,
        built: Dict[str, Any],
        top_k_per_garment: int = 3,
        remove_duplicates: bool = True,
        scene_image: Optional[Image.Image] = None,
    ) -> List[Dict]:
        """Search and rerank the catalog for every query produced by build_queries."""
        garment_results = []
        global_used_pids = set()
        w, h = built["size"]

        full_emb = built.get("full_embedding")
        if full_emb is not None:
            full_hits = self.catalog_index.search(full_emb, top_k=max(top_k_per_garment, 30))
            if full_hits and full_hits[0][1] >= 0.93:
                top = full_hits[:top_k_per_garment]
                global_used_pids.update(pid for pid, _ in top)
                garment_results.append({
                    "garment_label": "Whole Image",
                    "item_name": "Whole Image",
                    "region": "Full Image",
                    "box": (0, 0, w, h),
                    "crop_image": scene_image,
                    "confidence": float(full_hits[0][1]),
                    "dominant_color": "",
                    "pattern": "",
                    "attributes": {},
                    "icon": "🖼️",
                    "is_uncertain": False,
                    "embedding": full_emb,
                    "recommendations": [(pid, sim, sim) for pid, sim in top],
                    "raw_candidates": full_hits,
                    "diagnostics": [],
                })

        for q in built["queries"]:
            raw_candidates = self.catalog_index.search_candidates(
                q["embedding"], target_category=q["label"],
                candidate_pool_size=CANDIDATE_POOL_SIZE, target_item=q["item_name"],
            )

            candidates = [c for c in raw_candidates if not (remove_duplicates and c[0] in global_used_pids)]
            if len(candidates) < top_k_per_garment:
                candidates = raw_candidates

            reranked = self.rerank_candidates(
                raw_candidates=candidates,
                query_category=q["label"],
                query_color=q["dominant_color"],
                query_pattern=q["pattern"],
                top_k=top_k_per_garment,
                query_item=q["item_name"],
                query_hist=q["color_hist"],
            )
            global_used_pids.update(item[0] for item in reranked)

            garment_results.append({
                "garment_label": q["label"],
                "item_name": q["item_name"],
                "region": q["region"],
                "box": q["box"],
                "crop_image": q.get("crop_image"),
                "confidence": q["confidence"],
                "dominant_color": q["dominant_color"],
                "pattern": q["pattern"],
                "attributes": {},
                "icon": q["icon"],
                "is_uncertain": q["is_uncertain"],
                "embedding": q["embedding"],
                "color_hist": q["color_hist"],
                "recommendations": [(item[0], item[1], item[2]) for item in reranked],
                "raw_candidates": raw_candidates,
                "diagnostics": [item[3] for item in reranked],
            })

        return garment_results

    def retrieve_garments(
        self,
        scene_image: Image.Image,
        top_k_per_garment: int = 3,
        remove_duplicates: bool = True,
    ) -> List[Dict]:
        """Detect the items in a photo and return ranked catalog products for each one."""
        if self.catalog_index.get_size() == 0:
            return []
        built = self.build_queries(scene_image)
        return self.match_queries(built, top_k_per_garment, remove_duplicates, scene_image=scene_image)

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
