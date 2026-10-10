import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Ensembled prompts for fashion categories
GARMENT_TAXONOMY = {
    "Dress": [
        "a photo of a dress",
        "a woman wearing a dress",
        "a one-piece fashion dress",
        "a summer dress",
        "an evening gown or dress",
        "a midi or maxi dress",
        "a casual dress",
    ],
    "Top / Shirt": [
        "a photo of a top or blouse",
        "a shirt or t-shirt",
        "a casual blouse",
        "a woman wearing a shirt",
        "a knit sweater or top",
    ],
    "Pants / Trousers": [
        "a photo of pants",
        "trousers or slacks",
        "denim jeans",
        "a woman wearing pants",
    ],
    "Skirt": [
        "a photo of a skirt",
        "a pleated skirt",
        "a mini skirt",
        "a pencil skirt",
        "a woman wearing a skirt",
    ],
    "Jacket / Coat": [
        "a photo of a jacket",
        "a blazer",
        "a winter coat",
        "outerwear jacket",
    ],
    "Footwear": [
        "a photo of shoes",
        "high heels or pumps",
        "sneakers or trainers",
        "leather boots",
        "sandals",
    ],
    "Bag / Handbag": [
        "a photo of a handbag or purse",
        "a tote bag",
        "a shoulder bag",
        "a leather bag",
    ],
}

# Separate prompts for Dress vs Two-Piece Outfit discrimination
DRESS_PROMPTS = [
    "a photo of a woman wearing a dress",
    "a photo of a one-piece dress",
    "a stylish dress outfit",
    "an elegant dress",
]

SEPARATES_PROMPTS = [
    "a photo of a woman in a top and pants",
    "a photo of a shirt and trousers",
    "a photo of a blouse and skirt",
    "a two-piece top and bottom outfit",
]


class GarmentDetector:
    """Detects, segments, and classifies garments (especially dresses, tops, bottoms) in fashion scenes.

    Uses Faster R-CNN for person/apparel localization combined with calibrated OpenCLIP
    zero-shot classification to accurately distinguish dresses from two-piece outfits.
    """

    def __init__(self, clip_encoder=None, use_torchvision_detector: bool = True):
        self.clip_encoder = clip_encoder
        self.detection_model = None
        self.device = getattr(clip_encoder, "device", torch.device("cpu"))

        self.category_names = list(GARMENT_TAXONOMY.keys())
        self.category_embeddings = None
        self.dress_feat = None
        self.separates_feat = None
        self.prior_bias = None

        if self.clip_encoder is not None:
            self._precompute_text_prompts()

        if use_torchvision_detector:
            self._init_detector()

    def _precompute_text_prompts(self):
        """Precompute ensembled and calibrated text embeddings for all garment categories."""
        try:
            import open_clip
            tokenizer = open_clip.get_tokenizer(self.clip_encoder.model_name)

            # Category embeddings
            cat_embs = []
            for cat in self.category_names:
                toks = tokenizer(GARMENT_TAXONOMY[cat]).to(self.device)
                with torch.no_grad():
                    embs = self.clip_encoder.model.encode_text(toks)
                    embs = embs / embs.norm(dim=-1, keepdim=True)
                    mean_e = embs.mean(dim=0)
                    mean_e = mean_e / mean_e.norm(dim=-1, keepdim=True)
                    cat_embs.append(mean_e)
            self.category_embeddings = torch.stack(cat_embs)

            # Dress vs Separates discriminant embeddings
            with torch.no_grad():
                d_toks = tokenizer(DRESS_PROMPTS).to(self.device)
                d_embs = self.clip_encoder.model.encode_text(d_toks)
                d_embs = d_embs / d_embs.norm(dim=-1, keepdim=True)
                self.dress_feat = d_embs.mean(dim=0)
                self.dress_feat = self.dress_feat / self.dress_feat.norm(dim=-1, keepdim=True)

                s_toks = tokenizer(SEPARATES_PROMPTS).to(self.device)
                s_embs = self.clip_encoder.model.encode_text(s_toks)
                s_embs = s_embs / s_embs.norm(dim=-1, keepdim=True)
                self.separates_feat = s_embs.mean(dim=0)
                self.separates_feat = self.separates_feat / self.separates_feat.norm(dim=-1, keepdim=True)

                # Background prior for calibration
                bg_toks = tokenizer(["a photo of clothing", "a fashion photo", "apparel"]).to(self.device)
                bg_emb = self.clip_encoder.model.encode_text(bg_toks).mean(dim=0)
                bg_emb = bg_emb / bg_emb.norm(dim=-1, keepdim=True)
                self.prior_bias = (bg_emb @ self.category_embeddings.T)

            logger.info("Garment taxonomy text embeddings precomputed and calibrated.")
        except Exception as e:
            logger.warning(f"Could not precompute text prompts: {e}")

    def _init_detector(self):
        """Initialize lightweight torchvision Faster R-CNN MobileNet V3 for person localization."""
        try:
            import torchvision.models.detection as d
            weights = d.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self.detection_model = d.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(self.device)
            self.detection_model.eval()
            logger.info("Torchvision Faster R-CNN detector initialized.")
        except Exception as e:
            logger.info(f"Torchvision detector fallback: {e}")
            self.detection_model = None

    def _detect_persons(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect person bounding boxes (COCO class 1)."""
        if self.detection_model is None:
            return []

        try:
            import torchvision.transforms.functional as F
            tensor_img = F.to_tensor(image.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                predictions = self.detection_model(tensor_img)[0]

            boxes = predictions["boxes"].cpu().numpy()
            labels = predictions["labels"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy()

            person_boxes = []
            for box, label, score in zip(boxes, labels, scores):
                if label == 1 and score > 0.35:
                    x1, y1, x2, y2 = map(int, box)
                    if (x2 - x1) > 40 and (y2 - y1) > 80:
                        person_boxes.append((x1, y1, x2, y2))

            # Sort by area (largest person first)
            person_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            return person_boxes
        except Exception as e:
            logger.debug(f"Person detection failed: {e}")
            return []

    def classify_crop(self, crop_image: Image.Image, candidate_categories: Optional[List[str]] = None) -> Tuple[str, float]:
        """Classify a garment crop into fashion categories with bias calibration."""
        if self.clip_encoder is None or self.category_embeddings is None:
            return ("Apparel", 1.0)

        try:
            crop_rgb = crop_image.convert("RGB")
            with torch.no_grad():
                t = self.clip_encoder.preprocess(crop_rgb).unsqueeze(0).to(self.device)
                f = self.clip_encoder.model.encode_image(t)
                f = f / f.norm(dim=-1, keepdim=True)

                sims = (f @ self.category_embeddings.T)[0]
                if self.prior_bias is not None:
                    sims = sims - self.prior_bias * 0.5

                if candidate_categories:
                    cand_indices = [self.category_names.index(c) for c in candidate_categories if c in self.category_names]
                    sub_sims = sims[cand_indices]
                    probs = torch.softmax(sub_sims * 15.0, dim=-1)
                    best_idx = int(torch.argmax(probs))
                    return (candidate_categories[best_idx], float(probs[best_idx]))
                else:
                    probs = torch.softmax(sims * 15.0, dim=-1)
                    best_idx = int(torch.argmax(probs))
                    return (self.category_names[best_idx], float(probs[best_idx]))
        except Exception as e:
            logger.debug(f"Classification failed: {e}")
            return ("Apparel", 1.0)

    def is_dress_outfit(self, outfit_crop: Image.Image) -> Tuple[bool, float]:
        """Determine if the outfit is a one-piece dress vs separate top and bottom."""
        if self.clip_encoder is None or self.dress_feat is None or self.separates_feat is None:
            return (False, 0.5)

        try:
            with torch.no_grad():
                t = self.clip_encoder.preprocess(outfit_crop.convert("RGB")).unsqueeze(0).to(self.device)
                f = self.clip_encoder.model.encode_image(t)
                f = f / f.norm(dim=-1, keepdim=True)

                d_sim = (f @ self.dress_feat.unsqueeze(-1))[0, 0].item()
                s_sim = (f @ self.separates_feat.unsqueeze(-1))[0, 0].item()

                probs = torch.softmax(torch.tensor([d_sim, s_sim]) * 20.0, dim=-1)
                is_dress = d_sim >= (s_sim - 0.015)  # slight tie-breaker for full outfit dresses
                conf = float(probs[0] if is_dress else probs[1])
                return (bool(is_dress), conf)
        except Exception as e:
            logger.debug(f"Dress check failed: {e}")
            return (False, 0.5)

    def extract_garments(self, scene_image: Image.Image) -> List[Dict]:
        """Extract localized garment regions from a scene image.

        Accurately isolates:
        - 👗 Dress (if one-piece outfit detected)
        - 👚 Top / Shirt
        - 👖 Pants / Skirt
        - 👠 Footwear
        """
        w, h = scene_image.size
        crops = []

        person_boxes = self._detect_persons(scene_image)

        if person_boxes:
            px1, py1, px2, py2 = person_boxes[0]
            pw = px2 - px1
            ph = py2 - py1

            # Key anatomical regions
            full_outfit_box = (
                max(0, px1 - int(pw * 0.05)),
                max(0, int(py1 + 0.10 * ph)),
                min(w, px2 + int(pw * 0.05)),
                min(h, int(py1 + 0.88 * ph))
            )
            upper_box = (
                max(0, px1 - int(pw * 0.05)),
                max(0, int(py1 + 0.08 * ph)),
                min(w, px2 + int(pw * 0.05)),
                min(h, int(py1 + 0.52 * ph))
            )
            lower_box = (
                max(0, px1 - int(pw * 0.05)),
                max(0, int(py1 + 0.48 * ph)),
                min(w, px2 + int(pw * 0.05)),
                min(h, int(py1 + 0.90 * ph))
            )
            shoes_box = (
                max(0, px1),
                max(0, int(py1 + 0.80 * ph)),
                min(w, px2),
                min(h, py2)
            )

            # Test if outfit is a dress
            outfit_crop = scene_image.crop(full_outfit_box)
            is_dress, dress_conf = self.is_dress_outfit(outfit_crop)

            if is_dress:
                crops.append({
                    "region": "Main Body",
                    "label": "Dress",
                    "box": full_outfit_box,
                    "image": outfit_crop,
                    "confidence": dress_conf,
                    "icon": "👗"
                })
            else:
                top_crop = scene_image.crop(upper_box)
                top_label, top_conf = self.classify_crop(top_crop, ["Top / Shirt", "Jacket / Coat"])
                crops.append({
                    "region": "Upper Body",
                    "label": top_label,
                    "box": upper_box,
                    "image": top_crop,
                    "confidence": top_conf,
                    "icon": "👚"
                })

                bot_crop = scene_image.crop(lower_box)
                bot_label, bot_conf = self.classify_crop(bot_crop, ["Pants / Trousers", "Skirt"])
                crops.append({
                    "region": "Lower Body",
                    "label": bot_label,
                    "box": lower_box,
                    "image": bot_crop,
                    "confidence": bot_conf,
                    "icon": "👖" if "Pants" in bot_label else "👗"
                })

            # Shoes crop if height is substantial
            if (shoes_box[3] - shoes_box[1]) > 20:
                shoe_crop = scene_image.crop(shoes_box)
                crops.append({
                    "region": "Footwear",
                    "label": "Footwear",
                    "box": shoes_box,
                    "image": shoe_crop,
                    "confidence": 0.82,
                    "icon": "👠"
                })

        else:
            # Fallback anatomical proposals across scene
            full_outfit_box = (int(0.12 * w), int(0.10 * h), int(0.88 * w), int(0.88 * h))
            outfit_crop = scene_image.crop(full_outfit_box)
            is_dress, dress_conf = self.is_dress_outfit(outfit_crop)

            if is_dress:
                crops.append({
                    "region": "Central Outfit",
                    "label": "Dress",
                    "box": full_outfit_box,
                    "image": outfit_crop,
                    "confidence": dress_conf,
                    "icon": "👗"
                })
            else:
                upper_box = (int(0.10 * w), int(0.08 * h), int(0.90 * w), int(0.55 * h))
                top_crop = scene_image.crop(upper_box)
                top_label, top_conf = self.classify_crop(top_crop, ["Top / Shirt", "Jacket / Coat"])
                crops.append({
                    "region": "Upper Region",
                    "label": top_label,
                    "box": upper_box,
                    "image": top_crop,
                    "confidence": top_conf,
                    "icon": "👚"
                })

                lower_box = (int(0.12 * w), int(0.48 * h), int(0.88 * w), int(0.90 * h))
                bot_crop = scene_image.crop(lower_box)
                bot_label, bot_conf = self.classify_crop(bot_crop, ["Pants / Trousers", "Skirt"])
                crops.append({
                    "region": "Lower Region",
                    "label": bot_label,
                    "box": lower_box,
                    "image": bot_crop,
                    "confidence": bot_conf,
                    "icon": "👖"
                })

        return crops

    def detect_dress(self, scene_image: Image.Image) -> Optional[Dict]:
        """Specifically isolate and extract the dress region from a fashion image."""
        w, h = scene_image.size
        person_boxes = self._detect_persons(scene_image)

        if person_boxes:
            px1, py1, px2, py2 = person_boxes[0]
            pw = px2 - px1
            ph = py2 - py1
            box = (
                max(0, px1 - int(pw * 0.05)),
                max(0, int(py1 + 0.10 * ph)),
                min(w, px2 + int(pw * 0.05)),
                min(h, int(py1 + 0.88 * ph))
            )
        else:
            box = (int(0.15 * w), int(0.12 * h), int(0.85 * w), int(0.88 * h))

        crop = scene_image.crop(box)
        is_dress, conf = self.is_dress_outfit(crop)

        return {
            "label": "Dress",
            "box": box,
            "crop_image": crop,
            "confidence": conf,
            "is_dress": is_dress
        }

    def draw_detections(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        """Annotate image with sleek colored bounding boxes and badges."""
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)

        palette = {
            "Dress": "#10b981",          # Emerald green
            "Top / Shirt": "#3b82f6",     # Blue
            "Pants / Trousers": "#8b5cf6",# Violet
            "Skirt": "#ec4899",           # Pink
            "Footwear": "#f59e0b",        # Amber
            "Jacket / Coat": "#06b6d4",   # Cyan
        }

        for det in detections:
            box = det.get("box")
            if not box:
                continue

            label = det.get("label", "Item")
            conf = det.get("confidence", 0.0)
            icon = det.get("icon", "🏷️")
            color = palette.get(label, "#3b82f6")

            # Box outline
            x1, y1, x2, y2 = box
            for offset in range(3):
                draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

            # Label badge
            text = f"{icon} {label} ({conf:.0%})"
            text_w = len(text) * 7 + 10
            draw.rectangle([x1, max(0, y1 - 22), x1 + text_w, y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - 19)), text, fill="#ffffff")

        return annotated
