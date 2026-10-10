import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Standard garment category taxonomy
GARMENT_CATEGORIES = [
    "top, shirt, t-shirt, or blouse",
    "jacket, coat, blazer, or outerwear",
    "dress or gown",
    "pants, trousers, jeans, or slacks",
    "skirt",
    "shoes, sneakers, boots, or footwear",
    "handbag, purse, tote, or backpack",
]

CATEGORY_LABELS = [
    "Top / Shirt",
    "Jacket / Coat",
    "Dress",
    "Pants / Trousers",
    "Skirt",
    "Shoes / Footwear",
    "Handbag / Bag",
]


class GarmentDetector:
    """Detects and extracts garment crops from fashion scene images.

    Combines person/salient region localization with OpenCLIP zero-shot
    clothing category classification to isolate garments (tops, trousers,
    jackets, dresses, bags, shoes).
    """

    def __init__(self, clip_encoder=None, use_torchvision_detector: bool = False):
        self.clip_encoder = clip_encoder
        self.detection_model = None
        self.device = getattr(clip_encoder, "device", torch.device("cpu"))

        # Precomputed text embeddings for zero-shot garment classification
        self.text_features = None
        if self.clip_encoder is not None:
            self._precompute_text_prompts()

        if use_torchvision_detector:
            self._init_detector()

    def _precompute_text_prompts(self):
        """Precompute text embeddings for garment categories."""
        try:
            import open_clip
            prompts = [f"a photo of a fashion {cat}" for cat in GARMENT_CATEGORIES]
            tokenizer = open_clip.get_tokenizer(self.clip_encoder.model_name)
            tokens = tokenizer(prompts).to(self.device)
            with torch.no_grad():
                text_emb = self.clip_encoder.model.encode_text(tokens)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                self.text_features = text_emb
            logger.info("Garment category text embeddings precomputed.")
        except Exception as e:
            logger.warning(f"Could not precompute text prompts: {e}")

    def _init_detector(self):
        """Attempt to load a lightweight torchvision Faster R-CNN for person/bag detection."""
        try:
            import torchvision.models.detection as d
            # Use mobilenet backbone for fast CPU inference if available, else standard
            if hasattr(d, "fasterrcnn_mobilenet_v3_large_320_fpn"):
                weights = d.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                self.detection_model = d.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(self.device)
            elif hasattr(d, "fasterrcnn_resnet50_fpn"):
                weights = d.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                self.detection_model = d.fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
            if self.detection_model is not None:
                self.detection_model.eval()
                logger.info("Torchvision object detector initialized.")
        except Exception as e:
            logger.info(f"Torchvision detector not initialized ({e}). Using anatomical fashion cropper.")
            self.detection_model = None

    def _detect_persons(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect person bounding boxes (COCO class 1)."""
        if self.detection_model is None:
            return []

        try:
            import torchvision.transforms.functional as F
            tensor_img = F.to_tensor(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                predictions = self.detection_model(tensor_img)[0]

            boxes = predictions["boxes"].cpu().numpy()
            labels = predictions["labels"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy()

            # Person is class 1 in COCO
            person_boxes = []
            for box, label, score in zip(boxes, labels, scores):
                if label == 1 and score > 0.4:
                    x1, y1, x2, y2 = map(int, box)
                    # Filter out tiny boxes
                    if (x2 - x1) > 50 and (y2 - y1) > 100:
                        person_boxes.append((x1, y1, x2, y2))
            return person_boxes
        except Exception as e:
            logger.debug(f"Person detection failed: {e}")
            return []

    def classify_crop(self, crop_image: Image.Image) -> Tuple[str, float]:
        """Classify a garment crop into standard garment categories using zero-shot CLIP.

        Args:
            crop_image: PIL Image of the cropped garment

        Returns:
            (category_label, confidence_score)
        """
        if self.clip_encoder is None or self.text_features is None:
            return ("General Apparel", 1.0)

        try:
            with torch.no_grad():
                crop_tensor = self.clip_encoder.preprocess(crop_image).unsqueeze(0).to(self.device)
                crop_emb = self.clip_encoder.model.encode_image(crop_tensor)
                crop_emb = crop_emb / crop_emb.norm(dim=-1, keepdim=True)

                # Similarity with categories
                logits = (crop_emb @ self.text_features.T)[0]
                probs = torch.softmax(logits * 100.0, dim=-1)
                best_idx = int(torch.argmax(probs))
                confidence = float(probs[best_idx])
                return (CATEGORY_LABELS[best_idx], confidence)
        except Exception as e:
            logger.debug(f"Crop classification failed: {e}")
            return ("General Apparel", 1.0)

    def extract_garments(self, scene_image: Image.Image) -> List[Dict]:
        """Extract garment proposals from a scene image.

        Returns a list of dicts:
        [{
            "label": "Top / Shirt",
            "region": "Upper Body",
            "box": (x1, y1, x2, y2),
            "image": PIL.Image,
            "confidence": 0.85
        }, ...]
        """
        w, h = scene_image.size
        crops = []

        # 1. Try object detector first
        person_boxes = self._detect_persons(scene_image)

        if person_boxes:
            # Use largest detected person
            person_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            px1, py1, px2, py2 = person_boxes[0]
            pw = px2 - px1
            ph = py2 - py1

            # Anatomical regions relative to detected person:
            # Upper body (Tops/Jackets): 10% to 55% of height
            upper_box = (px1, int(py1 + 0.10 * ph), px2, int(py1 + 0.55 * ph))
            # Lower body (Pants/Skirts): 45% to 85% of height
            lower_box = (px1, int(py1 + 0.45 * ph), px2, int(py1 + 0.88 * ph))
            # Full outfit (Dress/Outerwear): 10% to 90% of height
            full_body_box = (px1, int(py1 + 0.10 * ph), px2, int(py1 + 0.95 * ph))
            # Shoes: 80% to 100% of height
            shoes_box = (px1, int(py1 + 0.82 * ph), px2, py2)

            proposals = [
                ("Upper Body (Top / Jacket)", upper_box),
                ("Lower Body (Pants / Skirt)", lower_box),
                ("Full Outfit (Dress / Suit)", full_body_box),
                ("Footwear (Shoes / Boots)", shoes_box),
            ]
        else:
            # Anatomical fallback crops across the full scene
            proposals = [
                ("Upper Region (Tops / Outerwear)", (int(0.1 * w), int(0.05 * h), int(0.9 * w), int(0.55 * h))),
                ("Lower Region (Bottoms)", (int(0.15 * w), int(0.45 * h), int(0.85 * w), int(0.88 * h))),
                ("Center Outfit Focus", (int(0.1 * w), int(0.1 * h), int(0.9 * w), int(0.9 * h))),
                ("Lower Footwear Focus", (int(0.2 * w), int(0.75 * h), int(0.8 * w), int(0.98 * h))),
            ]

        for region_name, box in proposals:
            bx1, by1, bx2, by2 = box
            # Clamp to image boundaries
            bx1 = max(0, min(w - 1, bx1))
            by1 = max(0, min(h - 1, by1))
            bx2 = max(bx1 + 10, min(w, bx2))
            by2 = max(by1 + 10, min(h, by2))

            crop_img = scene_image.crop((bx1, by1, bx2, by2))
            category, confidence = self.classify_crop(crop_img)

            crops.append({
                "region": region_name,
                "label": category,
                "box": (bx1, by1, bx2, by2),
                "image": crop_img,
                "confidence": confidence,
            })

        return crops
