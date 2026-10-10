import logging
from typing import List, Dict, Optional, Tuple
import colorsys
import logging
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

logger = logging.getLogger(__name__)

CATEGORY_TAXONOMY = {
    "Jacket / Outerwear": [
        "a photo of a jacket", "an outerwear coat", "a black jacket", "a blazer",
        "a lightweight jacket", "a leather jacket", "a denim jacket", "a bomber jacket"
    ],
    "Top / T-shirt": [
        "a photo of a shirt or top", "a t-shirt", "a striped shirt or blouse",
        "a gathered top", "a casual top", "a button-up shirt", "a graphic t-shirt"
    ],
    "Pants / Trousers": [
        "a photo of pants", "a pair of trousers", "denim jeans", "blue jeans",
        "dark trousers", "black pants", "chinos", "slacks", "a person wearing jeans",
        "a person wearing pants", "a photo of jeans", "denim trousers"
    ],
    "Dress": [
        "a photo of a one-piece dress", "a fashion dress", "a summer dress",
        "an evening gown", "a cocktail dress", "a midi dress", "a full-length dress"
    ],
    "Skirt": [
        "a photo of a skirt", "a pleated skirt", "an a-line skirt",
        "a mini skirt", "a midi skirt", "a flared skirt"
    ],
    "Footwear": [
        "a photo of shoes", "a pair of footwear", "sneakers", "leather boots",
        "ankle boots", "sandals", "dress shoes"
    ],
    "Bag / Handbag": [
        "a photo of a handbag", "a purse", "a tote bag", "a backpack", "a shoulder bag"
    ],
    "Full Outfit / Look": [
        "a fashion outfit", "a complete look", "a stylish outfit", "a full dress outfit"
    ],
}

ANATOMICAL_ALLOWED_CATEGORIES = {
    "upper_body": ["Jacket / Outerwear", "Top / T-shirt"],
    "inner_chest": ["Top / T-shirt", "Jacket / Outerwear"],
    "full_outfit": ["Dress", "Full Outfit / Look"],
    "lower_body": ["Pants / Trousers", "Skirt"],
    "footwear": ["Footwear"],
    "general": list(CATEGORY_TAXONOMY.keys()),
}


class GarmentDetector:
    """Detects, isolates, and classifies individual garments from scene images.

    Combines torchvision Faster R-CNN object localization with anatomically constrained,
    calibrated OpenCLIP zero-shot classification to avoid false category assignments.
    """

    def __init__(self, clip_encoder=None, use_torchvision_detector: bool = True):
        self.clip_encoder = clip_encoder
        self.detection_model = None
        self.device = getattr(clip_encoder, "device", torch.device("cpu"))

        self.category_names = list(CATEGORY_TAXONOMY.keys())
        self.category_embeddings = None
        self.dress_feat = None
        self.separates_feat = None
        self.prior_bias = None

        if self.clip_encoder is not None:
            self._precompute_text_prompts()

        if use_torchvision_detector:
            self._init_detector()

    def _precompute_text_prompts(self):
        """Precompute category text embeddings for zero-shot classification."""
        try:
            import open_clip
            tokenizer = open_clip.get_tokenizer(self.clip_encoder.model_name)

            cat_embs = []
            for cat in self.category_names:
                toks = tokenizer(CATEGORY_TAXONOMY[cat]).to(self.device)
                with torch.no_grad():
                    embs = self.clip_encoder.model.encode_text(toks)
                    embs = embs / embs.norm(dim=-1, keepdim=True)
                    mean_e = embs.mean(dim=0)
                    mean_e = mean_e / mean_e.norm(dim=-1, keepdim=True)
                    cat_embs.append(mean_e)
            self.category_embeddings = torch.stack(cat_embs)

            # Dress vs Two-Piece discriminants
            d_toks = tokenizer(["a photo of a one-piece dress", "a woman wearing a dress"]).to(self.device)
            s_toks = tokenizer(["a photo of a jacket and shirt", "a person wearing a top and pants"]).to(self.device)
            with torch.no_grad():
                d_f = self.clip_encoder.model.encode_text(d_toks).mean(dim=0)
                self.dress_feat = d_f / d_f.norm(dim=-1, keepdim=True)
                s_f = self.clip_encoder.model.encode_text(s_toks).mean(dim=0)
                self.separates_feat = s_f / s_f.norm(dim=-1, keepdim=True)

                bg_toks = tokenizer(["a photo of clothing", "fashion apparel item"]).to(self.device)
                bg_emb = self.clip_encoder.model.encode_text(bg_toks).mean(dim=0)
                bg_emb = bg_emb / bg_emb.norm(dim=-1, keepdim=True)
                self.prior_bias = (bg_emb @ self.category_embeddings.T)

            logger.info("Garment taxonomy precomputed.")
        except Exception as e:
            logger.warning(f"Could not precompute prompts: {e}")

    def _init_detector(self):
        """Initialize Faster R-CNN MobileNet V3 object detector."""
        try:
            import torchvision.models.detection as d
            weights = d.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self.detection_model = d.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(self.device)
            self.detection_model.eval()
            logger.info("Faster R-CNN detector initialized successfully.")
        except Exception as e:
            logger.info(f"Torchvision detector fallback: {e}")
            self.detection_model = None

    def _detect_objects(
        self, image: Image.Image
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[Tuple[int, int, int, int], float, str]]]:
        """Detect person bounding boxes (COCO class 1) and handbag/backpack boxes (classes 31, 27)."""
        if self.detection_model is None:
            return [], []

        try:
            w, h = image.size
            import torchvision.transforms.functional as F
            tensor_img = F.to_tensor(image.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.detection_model(tensor_img)[0]

            boxes = preds["boxes"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()

            person_boxes = []
            bag_detections = []
            for box, label, score in zip(boxes, labels, scores):
                x1 = max(0, min(w - 1, int(box[0])))
                y1 = max(0, min(h - 1, int(box[1])))
                x2 = max(x1 + 1, min(w, int(box[2])))
                y2 = max(y1 + 1, min(h, int(box[3])))
                box_w = x2 - x1
                box_h = y2 - y1

                if label == 1 and score > 0.35:
                    if box_w > 40 and box_h > 80:
                        person_boxes.append((x1, y1, x2, y2))
                elif label in (31, 27) and score > 0.25:  # handbag or backpack
                    if box_w > 18 and box_h > 18:
                        bag_detections.append(((x1, y1, x2, y2), float(score), "Bag / Handbag"))

            # Largest person first
            person_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            # Highest confidence bag first
            bag_detections.sort(key=lambda b: b[1], reverse=True)
            return person_boxes, bag_detections
        except Exception as e:
            logger.debug(f"Object detection error: {e}")
            return [], []

    def _detect_persons(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect person bounding boxes (COCO class 1) for backward compatibility."""
        persons, _ = self._detect_objects(image)
        return persons

    def classify_crop(
        self, crop_image: Image.Image, candidate_categories: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """Classify a crop strictly within the provided candidate categories.

        Enforces anatomical constraints to prevent upper-body crops from being labeled as skirts or shoes.
        """
        if self.clip_encoder is None or self.category_embeddings is None:
            return (candidate_categories[0] if candidate_categories else "Apparel", 0.5)

        allowed = candidate_categories or self.category_names

        try:
            with torch.no_grad():
                t = self.clip_encoder.preprocess(crop_image.convert("RGB")).unsqueeze(0).to(self.device)
                f = self.clip_encoder.model.encode_image(t)
                f = f / f.norm(dim=-1, keepdim=True)

                sims = (f @ self.category_embeddings.T)[0]
                if self.prior_bias is not None:
                    sims = sims - self.prior_bias * 0.25

                cand_indices = [self.category_names.index(c) for c in allowed if c in self.category_names]
                sub_sims = sims[cand_indices]
                probs = torch.softmax(sub_sims * 15.0, dim=-1)
                best_sub_idx = int(torch.argmax(probs))
                return (allowed[best_sub_idx], float(probs[best_sub_idx]))
        except Exception as e:
            logger.debug(f"Crop classification failed: {e}")
            return (allowed[0], 0.5)

    def is_dress_outfit(self, outfit_crop: Image.Image) -> Tuple[bool, float]:
        """Check if an outfit is a one-piece dress vs separate pieces."""
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
                is_dress = d_sim > (s_sim + 0.03)  # Requires genuine dress margin
                return (bool(is_dress), float(probs[0] if is_dress else probs[1]))
        except Exception:
            return (False, 0.5)

    def extract_color(self, crop: Image.Image) -> str:
        """Extract dominant color for the detected garment crop with border subtraction and HSV quantization."""
        try:
            crop_rgb = crop.convert("RGB").resize((70, 70))
            arr = np.array(crop_rgb)
            h, w, _ = arr.shape

            # Sample border pixels (outer 5%) to estimate backdrop/ambient color
            borders = np.concatenate([
                arr[:4, :].reshape(-1, 3),
                arr[-4:, :].reshape(-1, 3),
                arr[:, :4].reshape(-1, 3),
                arr[:, -4:].reshape(-1, 3),
            ], axis=0)
            bg_color = np.median(borders, axis=0)

            pixels = arr.reshape(-1, 3).astype(float)
            dist_to_bg = np.linalg.norm(pixels - bg_color, axis=1)

            # Central 60% coordinates
            cx1, cx2 = int(w * 0.20), int(w * 0.80)
            cy1, cy2 = int(h * 0.20), int(h * 0.80)
            center_pixels = arr[cy1:cy2, cx1:cx2].reshape(-1, 3)

            # Filter background if distinct
            is_bg = (dist_to_bg < 32) | (
                (pixels[:, 0] > 230) & (pixels[:, 1] > 230) & (pixels[:, 2] > 230) & (np.ptp(pixels, axis=1) < 18)
            )

            if is_bg.sum() > 0.90 * len(pixels):
                fg = center_pixels
            else:
                fg_candidates = pixels[~is_bg]
                fg = fg_candidates if len(fg_candidates) > 25 else center_pixels

            med_rgb = np.median(fg, axis=0)
            r, g, b = med_rgb
            h_norm, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h_deg = h_norm * 360.0

            if v < 0.24 or (v < 0.36 and s < 0.22):
                return "Black"
            if v > 0.88 and s < 0.12:
                return "White"
            if s < 0.16:
                return "Dark Grey" if v < 0.45 else ("Grey" if v < 0.72 else "Light Grey")

            if h_deg < 12 or h_deg >= 348:
                return "Maroon / Burgundy" if v < 0.50 else "Red"
            elif 12 <= h_deg < 42:
                return "Brown / Tan" if (v < 0.55 or s < 0.45) else "Orange"
            elif 42 <= h_deg < 70:
                return "Beige / Khaki" if s < 0.40 else "Yellow / Gold"
            elif 70 <= h_deg < 165:
                return "Olive / Dark Green" if v < 0.45 else "Green"
            elif 165 <= h_deg < 255:
                return "Navy / Dark Blue" if v < 0.45 else ("Light Blue" if (s < 0.45 and v > 0.7) else "Blue")
            elif 255 <= h_deg < 295:
                return "Purple"
            else:
                return "Pink"
        except Exception:
            return "Neutral"

    def extract_garments(self, scene_image: Image.Image) -> List[Dict]:
        """Extract garment proposals from scene image, checking the whole image and isolated items."""
        scene_image = ImageOps.exif_transpose(scene_image) or scene_image
        w, h = scene_image.size
        crops = []

        # 1. Check if the scene / outfit is a one-piece dress
        is_dress, dress_conf = self.is_dress_outfit(scene_image)
        dress_proposal = None
        if is_dress and dress_conf >= 0.58:
            dress_proposal = {
                "region": "Full Image",
                "label": "Dress",
                "box": (0, 0, w, h),
                "image": scene_image,
                "confidence": dress_conf,
                "dominant_color": self.extract_color(scene_image),
                "icon": "👗",
                "is_uncertain": False,
            }

        person_boxes, bag_detections = self._detect_objects(scene_image)

        if person_boxes:
            px1, py1, px2, py2 = person_boxes[0]
            pw = px2 - px1
            ph = py2 - py1
            person_ar = ph / max(1, pw)

            trousers_sufficiently_visible = (person_ar >= 1.25)
            footwear_sufficiently_visible = (py2 >= int(h * 0.70)) and (person_ar >= 1.20)

            # Check if isolated person region is a one-piece dress
            if not dress_proposal:
                full_outfit_box = (
                    max(0, px1 - int(pw * 0.05)),
                    max(0, int(py1 + 0.08 * ph)),
                    min(w, px2 + int(pw * 0.05)),
                    min(h, int(py1 + 0.90 * ph))
                )
                outfit_crop = scene_image.crop(full_outfit_box)
                person_is_dress, person_dress_conf = self.is_dress_outfit(outfit_crop)
                if person_is_dress and person_dress_conf >= 0.65:
                    dress_proposal = {
                        "region": "Full Outfit",
                        "label": "Dress",
                        "box": full_outfit_box,
                        "image": outfit_crop,
                        "confidence": person_dress_conf,
                        "dominant_color": self.extract_color(outfit_crop),
                        "icon": "👗",
                        "is_uncertain": False,
                    }

            if dress_proposal:
                crops.append(dress_proposal)
            else:
                crops.append({
                    "region": "Full Image",
                    "label": "Full Outfit / Look",
                    "box": (0, 0, w, h),
                    "image": scene_image,
                    "confidence": 0.90,
                    "dominant_color": self.extract_color(scene_image),
                    "icon": "✨",
                    "is_uncertain": False,
                })
                upper_span = 0.58 if trousers_sufficiently_visible else 0.88
                jacket_box = (
                    max(0, px1 - int(pw * 0.05)),
                    max(0, int(py1 + 0.08 * ph)),
                    min(w, px2 + int(pw * 0.05)),
                    min(h, int(py1 + upper_span * ph))
                )
                jacket_label = "Top / T-shirt"
                if (jacket_box[2] - jacket_box[0] > 30) and (jacket_box[3] - jacket_box[1] > 30):
                    jacket_crop = scene_image.crop(jacket_box)
                    jacket_label, jacket_conf = self.classify_crop(
                        jacket_crop, ANATOMICAL_ALLOWED_CATEGORIES["upper_body"]
                    )
                    jacket_color = self.extract_color(jacket_crop)
                    is_jacket = ("Jacket" in jacket_label)
                    crops.append({
                        "region": "Upper Outerwear" if is_jacket else "Shirt / Top",
                        "label": jacket_label,
                        "box": jacket_box,
                        "image": jacket_crop,
                        "confidence": jacket_conf,
                        "dominant_color": jacket_color,
                        "icon": "🧥" if is_jacket else "👚",
                        "is_uncertain": jacket_conf < 0.45,
                    })

                if jacket_label == "Jacket / Outerwear":
                    tshirt_box = (
                        max(0, int(px1 + 0.15 * pw)),
                        max(0, int(py1 + 0.12 * ph)),
                        min(w, int(px2 - 0.15 * pw)),
                        min(h, int(py1 + (0.52 * ph if trousers_sufficiently_visible else 0.75 * ph)))
                    )
                    if (tshirt_box[2] - tshirt_box[0] > 25) and (tshirt_box[3] - tshirt_box[1] > 25):
                        tshirt_crop = scene_image.crop(tshirt_box)
                        crops.append({
                            "region": "Inner Torso",
                            "label": "Top / T-shirt",
                            "box": tshirt_box,
                            "image": tshirt_crop,
                            "confidence": 0.88,
                            "dominant_color": self.extract_color(tshirt_crop),
                            "icon": "👕",
                            "is_uncertain": False,
                        })

                # Lower body: strict Skirt confidence threshold to prevent pants misclassification
                if person_ar >= 1.25:
                    lower_box = (
                        max(0, px1),
                        max(0, int(py1 + 0.40 * ph)),
                        min(w, px2),
                        min(h, int(py1 + (0.92 * ph if person_ar >= 1.55 else 0.86 * ph)))
                    )
                    if (lower_box[3] - lower_box[1]) > 35:
                        lower_crop = scene_image.crop(lower_box)
                        lower_label, lower_conf = self.classify_crop(
                            lower_crop, ANATOMICAL_ALLOWED_CATEGORIES["lower_body"]
                        )
                        is_skirt = (lower_label == "Skirt" and lower_conf >= 0.82)
                        if is_skirt:
                            skirt_box = (
                                max(0, px1),
                                max(0, int(py1 + 0.40 * ph)),
                                min(w, px2),
                                min(h, int(py1 + 0.78 * ph))
                            )
                            skirt_crop = scene_image.crop(skirt_box)
                            crops.append({
                                "region": "Lower Body",
                                "label": "Skirt",
                                "box": skirt_box,
                                "image": skirt_crop,
                                "confidence": lower_conf,
                                "dominant_color": self.extract_color(skirt_crop),
                                "icon": "👗",
                                "is_uncertain": False,
                            })
                        elif trousers_sufficiently_visible:
                            crops.append({
                                "region": "Lower Body",
                                "label": "Pants / Trousers",
                                "box": lower_box,
                                "image": lower_crop,
                                "confidence": lower_conf if lower_label == "Pants / Trousers" else (1.0 - lower_conf),
                                "dominant_color": self.extract_color(lower_crop),
                                "icon": "👖",
                                "is_uncertain": False,
                            })

            if footwear_sufficiently_visible:
                shoes_box = (
                    max(0, px1),
                    max(0, int(py1 + 0.82 * ph)),
                    min(w, px2),
                    min(h, int(py2 + 0.05 * ph))
                )
                if (shoes_box[3] - shoes_box[1]) > 18 and (shoes_box[2] - shoes_box[0]) > 20:
                    shoe_crop = scene_image.crop(shoes_box)
                    shoe_label, shoe_conf = self.classify_crop(shoe_crop, ["Footwear", "Pants / Trousers"])
                    if shoe_label == "Footwear" or shoe_conf >= 0.38:
                        crops.append({
                            "region": "Footwear",
                            "label": "Footwear",
                            "box": shoes_box,
                            "image": shoe_crop,
                            "confidence": shoe_conf,
                            "dominant_color": self.extract_color(shoe_crop),
                            "icon": "👠",
                            "is_uncertain": False,
                        })

        else:
            scene_ar = h / max(1, w)
            upper_box = (int(0.08 * w), int(0.06 * h), int(0.92 * w), int(0.58 * h))
            upper_crop = scene_image.crop(upper_box)
            upper_label, upper_conf = self.classify_crop(upper_crop, ["Jacket / Outerwear", "Top / T-shirt"])
            is_jacket = ("Jacket" in upper_label)
            crops.append({
                "region": "Upper Region" if is_jacket else "Shirt / Top",
                "label": upper_label,
                "box": upper_box,
                "image": upper_crop,
                "confidence": upper_conf,
                "dominant_color": self.extract_color(upper_crop),
                "icon": "🧥" if is_jacket else "👚",
                "is_uncertain": False,
            })

            if is_jacket:
                inner_box = (int(0.22 * w), int(0.12 * h), int(0.78 * w), int(0.48 * h))
                inner_crop = scene_image.crop(inner_box)
                crops.append({
                    "region": "Inner Torso",
                    "label": "Top / T-shirt",
                    "box": inner_box,
                    "image": inner_crop,
                    "confidence": 0.85,
                    "dominant_color": self.extract_color(inner_crop),
                    "icon": "👕",
                    "is_uncertain": False,
                })

            if scene_ar >= 1.25:
                lower_box = (int(0.12 * w), int(0.45 * h), int(0.88 * w), int(0.92 * h))
                lower_crop = scene_image.crop(lower_box)
                lower_label, lower_conf = self.classify_crop(lower_crop, ["Pants / Trousers", "Skirt"])
                is_skirt = (lower_label == "Skirt" and lower_conf >= 0.82)
                crops.append({
                    "region": "Lower Region",
                    "label": "Skirt" if is_skirt else "Pants / Trousers",
                    "box": lower_box,
                    "image": lower_crop,
                    "confidence": lower_conf,
                    "dominant_color": self.extract_color(lower_crop),
                    "icon": "👗" if is_skirt else "👖",
                    "is_uncertain": False,
                })

        # E: Detected Handbags / Bags from COCO detector
        for bag_box, bag_score, bag_label in bag_detections[:2]:
            bx1, by1, bx2, by2 = bag_box
            pad = 6
            padded_box = (
                max(0, bx1 - pad),
                max(0, by1 - pad),
                min(w, bx2 + pad),
                min(h, by2 + pad),
            )
            bag_crop = scene_image.crop(padded_box)
            crops.append({
                "region": "Handbag / Accessory",
                "label": "Bag / Handbag",
                "box": padded_box,
                "image": bag_crop,
                "confidence": bag_score,
                "dominant_color": self.extract_color(bag_crop),
                "icon": "👜",
                "is_uncertain": False,
            })

        return crops


    def draw_detections(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        """Annotate image with sleek colored bounding boxes and badges."""
        image = ImageOps.exif_transpose(image) or image
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        w, h = annotated.size

        palette = {
            "Jacket / Outerwear": "#2563eb",  # Royal Blue
            "Top / T-shirt": "#0284c7",       # Sky Blue
            "Pants / Trousers": "#7c3aed",   # Purple
            "Dress": "#059669",              # Emerald Green
            "Skirt": "#db2777",              # Pink
            "Footwear": "#d97706",           # Amber
            "Bag / Handbag": "#475569",      # Slate
        }

        for det in detections:
            box = det.get("box")
            if not box:
                continue

            label = det.get("label", "Item")
            conf = det.get("confidence", 0.0)
            color_name = det.get("dominant_color", "")
            icon = det.get("icon", "🏷️")
            border_color = palette.get(label, "#2563eb")

            x1, y1, x2, y2 = box
            x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
            x2, y2 = max(x1 + 1, min(w, x2)), max(y1 + 1, min(h, y2))

            for offset in range(3):
                draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=border_color)

            # Badge with Icon, Label, and Detected Color
            text = f"{icon} {label}"
            if color_name and color_name != "Neutral":
                text += f" ({color_name})"
            text += f" {conf:.0%}"

            text_w = min(w - x1, len(text) * 7 + 10)
            badge_y1 = max(0, y1 - 22)
            draw.rectangle([x1, badge_y1, x1 + text_w, badge_y1 + 22], fill=border_color)
            draw.text((x1 + 4, badge_y1 + 3), text, fill="#ffffff")

        return annotated
