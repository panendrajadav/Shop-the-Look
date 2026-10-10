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

# Specific item names within each category (name, CLIP prompt)
ITEM_NAMES = {
    "Jacket / Outerwear": [
        ("Blazer", "a blazer"), ("Denim Jacket", "a denim jacket"), ("Leather Jacket", "a leather jacket"),
        ("Coat", "a long coat"), ("Cardigan", "a knitted cardigan"), ("Hoodie", "a hoodie"),
        ("Bomber Jacket", "a bomber jacket"), ("Puffer Jacket", "a puffer jacket"),
    ],
    "Top / T-shirt": [
        ("T-shirt", "a t-shirt"), ("Shirt", "a button-up shirt"), ("Blouse", "a blouse"),
        ("Sweater", "a knitted sweater"), ("Tank Top", "a sleeveless tank top"), ("Crop Top", "a crop top"),
        ("Sweatshirt", "a sweatshirt"),
    ],
    "Pants / Trousers": [
        ("Jeans", "denim jeans"), ("Trousers", "tailored trousers"), ("Shorts", "shorts"),
        ("Leggings", "tight leggings"), ("Joggers", "jogger sweatpants"), ("Wide-leg Pants", "wide-leg pants"),
    ],
    "Dress": [
        ("Maxi Dress", "a long maxi dress"), ("Midi Dress", "a midi dress"), ("Mini Dress", "a short mini dress"),
        ("Gown", "an evening gown"), ("Shirt Dress", "a shirt dress"), ("Jumpsuit", "a jumpsuit"),
    ],
    "Skirt": [
        ("Mini Skirt", "a mini skirt"), ("Midi Skirt", "a midi skirt"), ("Maxi Skirt", "a long maxi skirt"),
        ("Pleated Skirt", "a pleated skirt"), ("Pencil Skirt", "a pencil skirt"), ("Denim Skirt", "a denim skirt"),
    ],
    "Footwear": [
        ("Sneakers", "sneakers"), ("Boots", "boots"), ("High Heels", "high heel shoes"),
        ("Sandals", "sandals"), ("Flats", "flat shoes"), ("Loafers", "loafers"),
    ],
    "Bag / Handbag": [
        ("Tote Bag", "a tote bag"), ("Shoulder Bag", "a shoulder bag"), ("Backpack", "a backpack"),
        ("Clutch", "a clutch bag"), ("Crossbody Bag", "a crossbody bag"),
    ],
    "Accessory": [
        ("Sunglasses", "sunglasses"), ("Necklace", "a necklace"), ("Earrings", "earrings"),
        ("Bracelet", "a bracelet"), ("Watch", "a wrist watch"), ("Hat", "a hat"),
        ("Belt", "a belt"), ("Scarf", "a scarf"), ("Ring", "a ring"),
    ],
}

# Same pattern vocabulary as the catalog metadata (scripts/enrich_catalog_metadata.py)
PATTERN_PROMPTS = {
    "Solid Color": "a photo of solid single color plain fabric",
    "Graphic / Printed": "a photo of graphic printed or logo fabric",
    "Striped": "a photo of striped fabric pattern",
    "Floral / Patterned": "a photo of floral or patterned decorative fabric",
    "Plaid / Checkered": "a photo of plaid or checkered pattern fabric",
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

            self.item_name_embeddings = {}
            with torch.no_grad():
                for cat, items in ITEM_NAMES.items():
                    e = self.clip_encoder.model.encode_text(
                        tokenizer([f"a photo of {p}" for _, p in items]).to(self.device)
                    )
                    self.item_name_embeddings[cat] = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()

            with torch.no_grad():
                e = self.clip_encoder.model.encode_text(tokenizer([
                    "a photo of a person wearing sunglasses",
                    "a photo of a face with no glasses",
                    "a photo of hair or the back of a head",
                    "a photo of a person wearing a hat",
                ]).to(self.device))
                self.eyewear_feats = e / e.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                e = self.clip_encoder.model.encode_text(tokenizer([
                    "a photo of a street, wall or building",
                    "a photo of grass, trees or flowers",
                    "a photo of a face or hair",
                    "a photo of a floor or pavement",
                    "a photo of a room or furniture",
                ]).to(self.device))
                self.background_feats = e / e.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                e = self.clip_encoder.model.encode_text(tokenizer(list(PATTERN_PROMPTS.values())).to(self.device))
                self.pattern_embeddings = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()

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
                elif label in (31, 27) and score > 0.25:  # handbag or backpack (CLIP-verified later)
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

    def name_item(self, crop_embedding: np.ndarray, category: str) -> str:
        """Return the specific item name (e.g. Jeans, Mini Skirt) for a crop within its category."""
        embs = getattr(self, "item_name_embeddings", {}).get(category)
        if embs is None or crop_embedding is None:
            return category
        return ITEM_NAMES[category][int(np.argmax(embs @ crop_embedding))][0]

    # Categories that can legitimately be seen in the same body region
    VERIFY_GROUPS = [
        {"Jacket / Outerwear", "Top / T-shirt", "Dress"},
        {"Pants / Trousers", "Skirt", "Dress"},
        {"Footwear"},
        {"Bag / Handbag"},
    ]

    def verify_crop(self, crop: Image.Image, label: str) -> bool:
        """Check that a region really shows the garment it was proposed as.

        Body regions are derived from the person box, so a region can land on something else
        (shoes in a close-up, background, a face). The crop is classified against every item
        category plus background prompts; it is kept only if its own category group wins, or
        comes close to the winner.
        """
        if self.clip_encoder is None or self.category_embeddings is None or getattr(self, "background_feats", None) is None:
            return True
        group = next((g for g in self.VERIFY_GROUPS if label in g), None)
        if group is None:
            return True
        try:
            with torch.no_grad():
                t = self.clip_encoder.preprocess(crop.convert("RGB")).unsqueeze(0).to(self.device)
                f = self.clip_encoder.model.encode_image(t)
                f = f / f.norm(dim=-1, keepdim=True)
                names = [c for c in self.category_names if c != "Full Outfit / Look"]
                idx = [self.category_names.index(c) for c in names]
                cat_sims = (f @ self.category_embeddings[idx].T)[0]
                bg_sim = (f @ self.background_feats.T)[0].max()
                probs = torch.softmax(torch.cat([cat_sims, bg_sim.view(1)]) * 100.0, dim=-1)
            own = float(sum(probs[i] for i, c in enumerate(names) if c in group))
            rivals = [float(probs[i]) for i, c in enumerate(names) if c not in group] + [float(probs[-1])]
            return own >= max(rivals) - 0.20
        except Exception:
            return True

    def sunglasses_confidence(self, head_crop: Image.Image) -> float:
        """Zero-shot probability that the head crop shows a person wearing sunglasses."""
        feats = getattr(self, "eyewear_feats", None)
        if self.clip_encoder is None or feats is None:
            return 0.0
        try:
            with torch.no_grad():
                t = self.clip_encoder.preprocess(head_crop.convert("RGB")).unsqueeze(0).to(self.device)
                f = self.clip_encoder.model.encode_image(t)
                f = f / f.norm(dim=-1, keepdim=True)
                return float(torch.softmax((f @ feats.T)[0] * 100.0, dim=-1)[0])
        except Exception:
            return 0.0

    def name_pattern(self, crop_embedding: np.ndarray) -> str:
        """Pattern of a crop (solid, striped, floral, ...), from the same prompts as the catalog."""
        embs = getattr(self, "pattern_embeddings", None)
        if embs is None or crop_embedding is None:
            return ""
        return list(PATTERN_PROMPTS.keys())[int(np.argmax(embs @ crop_embedding))]

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

        # A dress is judged on the detected person, not on the whole photo: backgrounds such
        # as flowers or curtains make whole-image dress checks fire on scenes with no dress.
        dress_proposal = None

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
                # Regions are kept apart so each crop holds one garment: the upper layer ends at the
                # waist (about half of a standing figure) and the lower body starts just below it.
                upper_span = 0.52 if trousers_sufficiently_visible else 0.88
                jacket_box = (
                    max(0, px1 - int(pw * 0.05)),
                    max(0, int(py1 + 0.12 * ph)),
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
                    # Inner shirt: the centre of the chest only, away from jacket fronts and sleeves
                    tshirt_box = (
                        max(0, int(px1 + 0.30 * pw)),
                        max(0, int(py1 + 0.16 * ph)),
                        min(w, int(px2 - 0.30 * pw)),
                        min(h, int(py1 + (0.46 * ph if trousers_sufficiently_visible else 0.70 * ph)))
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
                    # Legs sit inside the person box, whose width is set by arms and bags
                    lower_box = (
                        max(0, int(px1 + 0.08 * pw)),
                        max(0, int(py1 + 0.47 * ph)),
                        min(w, int(px2 - 0.08 * pw)),
                        min(h, int(py1 + (0.90 * ph if person_ar >= 1.55 else 0.86 * ph)))
                    )
                    if (lower_box[3] - lower_box[1]) > 35:
                        lower_crop = scene_image.crop(lower_box)
                        lower_label, lower_conf = self.classify_crop(
                            lower_crop, ANATOMICAL_ALLOWED_CATEGORIES["lower_body"]
                        )
                        is_skirt = (lower_label == "Skirt" and lower_conf >= 0.65)
                        if is_skirt:
                            skirt_box = (
                                max(0, int(px1 + 0.05 * pw)),
                                max(0, int(py1 + 0.45 * ph)),
                                min(w, int(px2 - 0.05 * pw)),
                                min(h, int(py1 + 0.76 * ph))
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

            # Sunglasses: zero-shot check on the head region of the person box
            head_box = (max(0, px1), max(0, py1), min(w, px2), min(h, int(py1 + 0.17 * ph)))
            if (head_box[2] - head_box[0]) > 24 and (head_box[3] - head_box[1]) > 24:
                head_crop = scene_image.crop(head_box)
                sun_conf = self.sunglasses_confidence(head_crop)
                if sun_conf >= 0.6:
                    crops.append({
                        "region": "Head",
                        "label": "Accessory",
                        "item_name": "Sunglasses",
                        "box": head_box,
                        "image": head_crop,
                        "confidence": sun_conf,
                        "dominant_color": self.extract_color(head_crop),
                        "icon": "🕶️",
                        "is_uncertain": False,
                    })

            if footwear_sufficiently_visible:
                shoes_box = (
                    max(0, px1),
                    max(0, int(py1 + 0.87 * ph)),
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
            # No person found: the photo shows an item on its own (a product shot or close-up),
            # so the whole image is one query, classified across all item categories.
            item_cats = [c for c in self.category_names if c != "Full Outfit / Look"]
            item_label, item_conf = self.classify_crop(scene_image, item_cats)
            crops.append({
                "region": "Full Image",
                "label": item_label,
                "box": (0, 0, w, h),
                "image": scene_image,
                "confidence": item_conf,
                "dominant_color": self.extract_color(scene_image),
                "icon": "🏷️",
                "is_uncertain": item_conf < 0.45,
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

        # Keep only regions that really show the proposed garment
        verified = [
            c for c in crops
            if c["label"] in ("Full Outfit / Look", "Accessory") or c["region"] == "Full Image"
            or self.verify_crop(c["image"], c["label"])
        ]
        if not any(c["label"] != "Full Outfit / Look" for c in verified):
            # Nothing on the "person" held up (often a false person detection on a close-up):
            # treat the photo as a single item instead.
            item_cats = [c for c in self.category_names if c != "Full Outfit / Look"]
            item_label, item_conf = self.classify_crop(scene_image, item_cats)
            return [{
                "region": "Full Image",
                "label": item_label,
                "box": (0, 0, w, h),
                "image": scene_image,
                "confidence": item_conf,
                "dominant_color": self.extract_color(scene_image),
                "icon": "\U0001F3F7\ufe0f",
                "is_uncertain": item_conf < 0.45,
            }]
        return verified


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

            category = det.get("label") or det.get("garment_label", "Item")
            if category == "Full Outfit / Look":
                continue  # the whole-photo query is not an item; drawing it only adds clutter
            label = det.get("item_name") or category
            conf = det.get("confidence", 0.0)
            color_name = det.get("dominant_color", "")
            border_color = palette.get(category, "#2563eb")

            x1, y1, x2, y2 = box
            x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
            x2, y2 = max(x1 + 1, min(w, x2)), max(y1 + 1, min(h, y2))

            for offset in range(3):
                draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=border_color)

            # Badge with Icon, Label, and Detected Color
            text = label  # the default PIL font cannot render emoji
            if color_name and color_name != "Neutral":
                text += f" ({color_name})"
            text += f" {conf:.0%}"

            text_w = min(w - x1, len(text) * 7 + 10)
            badge_y1 = max(0, y1 - 22)
            draw.rectangle([x1, badge_y1, x1 + text_w, badge_y1 + 22], fill=border_color)
            draw.text((x1 + 4, badge_y1 + 3), text, fill="#ffffff")

        return annotated
