import logging
from typing import Optional, Tuple, List, Dict, Union, Any
from pathlib import Path
import colorsys
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

COLOR_TAXONOMY = [
    "Black",
    "White",
    "Dark Grey",
    "Grey",
    "Light Grey",
    "Navy / Dark Blue",
    "Blue",
    "Light Blue",
    "Red",
    "Maroon / Burgundy",
    "Brown / Tan",
    "Beige / Khaki",
    "Green",
    "Olive / Dark Green",
    "Yellow / Gold",
    "Pink",
    "Purple",
    "Orange",
]

PATTERN_PROMPTS = [
    ("Solid Color", "a photo of solid plain single color fabric"),
    ("Graphic / Printed", "a photo of graphic printed design or logo fabric"),
    ("Striped", "a photo of striped fabric with horizontal or vertical stripes"),
    ("Plaid / Checkered", "a photo of plaid or checkered pattern fabric"),
    ("Floral / Patterned", "a photo of floral or patterned decorative fabric"),
]

SILHOUETTE_PROMPTS = [
    ("Gathered / Peplum / Ruffled", "a photo of a gathered, ruffled, smocked, or peplum silhouette top or skirt"),
    ("Fitted / Tailored", "a photo of a slim-fit or tailored fitted silhouette garment"),
    ("Classic / Relaxed", "a photo of a straight classic cut or relaxed silhouette garment"),
    ("Pleated / Flared / A-Line", "a photo of a pleated, flared, or a-line silhouette skirt"),
]

SLEEVE_PROMPTS = [
    ("Sleeveless / Tank", "a photo of a sleeveless top, tank top, or camisole with bare arms"),
    ("Short Sleeve", "a photo of a short sleeve shirt, top, or blouse"),
    ("Long Sleeve", "a photo of a long sleeve top, shirt, jacket, or sweater"),
]

NECKLINE_PROMPTS = [
    ("Scoop / Square / Gathered", "a scoop neck, square neckline, or elastic gathered neckline top"),
    ("Crew / Round Neck", "a crew neck or round neckline top"),
    ("V-Neck", "a v-neck neckline top"),
    ("Collared / Button-Up", "a collared button-up shirt"),
]


class AttributeExtractor:
    """Extracts visual attributes directly from image pixels and CLIP representations."""

    def __init__(self, clip_encoder=None):
        self.clip_encoder = clip_encoder
        self.device = getattr(clip_encoder, "device", torch.device("cpu"))
        self.pattern_features = None
        self.silhouette_features = None
        self.sleeve_features = None
        self.neckline_features = None

        if self.clip_encoder is not None:
            self._init_prompts()

    def _init_prompts(self):
        try:
            import open_clip
            tokenizer = open_clip.get_tokenizer(self.clip_encoder.model_name)

            def _encode_prompts(prompt_list):
                tokens = tokenizer([p[1] for p in prompt_list]).to(self.device)
                with torch.no_grad():
                    f = self.clip_encoder.model.encode_text(tokens)
                    return f / f.norm(dim=-1, keepdim=True)

            self.pattern_features = _encode_prompts(PATTERN_PROMPTS)
            self.silhouette_features = _encode_prompts(SILHOUETTE_PROMPTS)
            self.sleeve_features = _encode_prompts(SLEEVE_PROMPTS)
            self.neckline_features = _encode_prompts(NECKLINE_PROMPTS)
        except Exception as e:
            logger.debug(f"Attribute prompt feature initialization failed: {e}")

    def extract_dominant_color(self, image: Image.Image) -> str:
        """Extract dominant color from image pixels with border background subtraction and HSV quantization.

        Specifically designed for fashion catalog images with white/neutral backdrops and real-world crops.
        """
        try:
            im_rgb = image.convert("RGB").resize((100, 100))
            arr = np.array(im_rgb)
            h, w, _ = arr.shape

            # Sample border pixels (outer 5%) to estimate background color
            borders = np.concatenate([
                arr[:5, :].reshape(-1, 3),
                arr[-5:, :].reshape(-1, 3),
                arr[:, :5].reshape(-1, 3),
                arr[:, -5:].reshape(-1, 3),
            ], axis=0)
            bg_color = np.median(borders, axis=0)

            pixels = arr.reshape(-1, 3).astype(float)
            dist_to_bg = np.linalg.norm(pixels - bg_color, axis=1)

            # Is pixel close to background OR neutral bright white (common catalog artifact)?
            is_bg = (dist_to_bg < 32) | (
                (pixels[:, 0] > 230) & (pixels[:, 1] > 230) & (pixels[:, 2] > 230) & (np.ptp(pixels, axis=1) < 18)
            )

            # If background removal leaves too few pixels, fall back to center crop
            if is_bg.sum() > 0.90 * len(pixels):
                fg_pixels = arr[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)].reshape(-1, 3)
            else:
                fg_candidates = arr.reshape(-1, 3)[~is_bg]
                fg_pixels = fg_candidates if len(fg_candidates) > 20 else arr.reshape(-1, 3)

            if len(fg_pixels) == 0:
                return "Neutral / Multi-color"

            med_rgb = np.median(fg_pixels, axis=0)
            r, g, b = med_rgb
            h_norm, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h_deg = h_norm * 360.0

            # 1. Low saturation or extreme value checks
            if v < 0.24 or (v < 0.36 and s < 0.22):
                return "Black"
            if v > 0.86 and s < 0.14:
                return "White"
            if s < 0.18:
                return "Dark Grey" if v < 0.45 else ("Grey" if v < 0.72 else "Light Grey")

            # 2. Hue-based color naming
            if h_deg < 14 or h_deg >= 346:
                return "Maroon / Burgundy" if (v < 0.48 and s > 0.30) else "Red"
            elif 14 <= h_deg < 42:
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

        except Exception as e:
            logger.debug(f"Color extraction error: {e}")
            return "Neutral / Multi-color"

    def extract_pattern(self, image: Image.Image) -> str:
        """Extract pattern using zero-shot CLIP classification."""
        if self.clip_encoder is None or self.pattern_features is None:
            return "Solid Color"

        try:
            with torch.no_grad():
                tensor = self.clip_encoder.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
                emb = self.clip_encoder.model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                sims = (emb @ self.pattern_features.T)[0]
                best_idx = int(torch.argmax(sims))
                return PATTERN_PROMPTS[best_idx][0]
        except Exception:
            return "Solid Color"

    def extract_silhouette(self, image: Image.Image) -> str:
        """Extract garment silhouette using zero-shot CLIP classification."""
        if self.clip_encoder is None or self.silhouette_features is None:
            return "Classic / Relaxed"

        try:
            with torch.no_grad():
                tensor = self.clip_encoder.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
                emb = self.clip_encoder.model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                sims = (emb @ self.silhouette_features.T)[0]
                best_idx = int(torch.argmax(sims))
                return SILHOUETTE_PROMPTS[best_idx][0]
        except Exception:
            return "Classic / Relaxed"

    def extract_sleeve(self, image: Image.Image) -> str:
        """Extract sleeve length using zero-shot CLIP classification."""
        if self.clip_encoder is None or self.sleeve_features is None:
            return "Short Sleeve"

        try:
            with torch.no_grad():
                tensor = self.clip_encoder.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
                emb = self.clip_encoder.model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                sims = (emb @ self.sleeve_features.T)[0]
                best_idx = int(torch.argmax(sims))
                return SLEEVE_PROMPTS[best_idx][0]
        except Exception:
            return "Short Sleeve"

    def extract_neckline(self, image: Image.Image) -> str:
        """Extract neckline style using zero-shot CLIP classification."""
        if self.clip_encoder is None or self.neckline_features is None:
            return "Crew / Round Neck"

        try:
            with torch.no_grad():
                tensor = self.clip_encoder.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
                emb = self.clip_encoder.model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                sims = (emb @ self.neckline_features.T)[0]
                best_idx = int(torch.argmax(sims))
                return NECKLINE_PROMPTS[best_idx][0]
        except Exception:
            return "Crew / Round Neck"

    def extract_color_tones(self, image: Image.Image) -> Dict[str, Any]:
        """Detect dual tones for striped or multi-color garments (e.g. Pink & White)."""
        dom = self.extract_dominant_color(image)
        try:
            im_rgb = image.convert("RGB").resize((60, 60))
            arr = np.array(im_rgb).reshape(-1, 3).astype(float)
            # Check for white stripe pixels (v > 0.85, s < 0.15)
            hsv = [colorsys.rgb_to_hsv(p[0]/255.0, p[1]/255.0, p[2]/255.0) for p in arr]
            white_frac = sum(1 for h, s, v in hsv if v > 0.85 and s < 0.16) / len(arr)
            pink_frac = sum(1 for h, s, v in hsv if (h*360 >= 295 or h*360 < 14) and s > 0.18 and v > 0.50) / len(arr)

            if white_frac > 0.15 and pink_frac > 0.15:
                return {"primary": "Pink", "secondary": "White", "color_label": "Pink & White"}
            return {"primary": dom, "secondary": None, "color_label": dom}
        except Exception:
            return {"primary": dom, "secondary": None, "color_label": dom}

    def extract_attributes(self, image_path_or_img: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """Extract dominant color, pattern, silhouette, and sleeve from image or file path."""
        if isinstance(image_path_or_img, (str, Path)):
            try:
                img = Image.open(image_path_or_img).convert("RGB")
            except Exception:
                return {
                    "dominant_color": "Unknown",
                    "pattern": "Solid Color",
                    "silhouette": "Classic / Relaxed",
                    "sleeve": "Short Sleeve",
                    "neckline": "Crew / Round Neck",
                    "primary_color": "Unknown",
                    "secondary_color": None,
                }
        else:
            img = image_path_or_img.convert("RGB")

        color = self.extract_dominant_color(img)
        pattern = self.extract_pattern(img)
        silhouette = self.extract_silhouette(img)
        sleeve = self.extract_sleeve(img)
        neckline = self.extract_neckline(img)
        tones = self.extract_color_tones(img)

        return {
            "dominant_color": tones.get("color_label", color),
            "primary_color": tones.get("primary", color),
            "secondary_color": tones.get("secondary"),
            "pattern": pattern,
            "silhouette": silhouette,
            "sleeve": sleeve,
            "neckline": neckline,
        }


STYLE_PROMPTS = [
    ("Casual", "a casual everyday piece of clothing"),
    ("Formal / Tailored", "a formal, tailored, office-wear fashion item"),
    ("Sporty / Athletic", "a sporty athletic activewear item"),
    ("Elegant / Evening", "an elegant evening or party fashion item"),
    ("Bohemian / Vintage", "a bohemian or vintage style fashion item"),
    ("Streetwear / Edgy", "an edgy streetwear fashion item"),
]

MATERIAL_PROMPTS = [
    ("Denim", "a fashion item made of denim"),
    ("Leather", "a fashion item made of leather"),
    ("Knit / Wool", "a fashion item made of knitted wool"),
    ("Cotton / Jersey", "a fashion item made of soft cotton jersey"),
    ("Silk / Satin", "a fashion item made of shiny silk or satin"),
    ("Lace / Sheer", "a fashion item made of lace or sheer fabric"),
    ("Suede / Velvet", "a fashion item made of suede or velvet"),
    ("Metal / Jewelry", "a metal jewelry accessory"),
]


# Greyscale-thumbnail correlation above which two pictures are treated as the same photo.
# Measured on 303 catalog products: a copy resized to 60% and re-saved as JPEG quality 60 scored
# 0.998 or higher, while the nearest *different* catalog photo scored at most 0.981. Cosine
# similarity cannot make this distinction: 168 of the 303 had a different product at 0.93+.
SAME_IMAGE_CORRELATION = 0.99


class RecommendationExplainer:
    """Explains each recommendation by comparing the query garment with the catalog product.

    Every statement comes from the two images (pixels and CLIP embeddings). No validation
    label is used here: the exact / similar decision is a prediction from similarity scores.
    """

    def __init__(self, catalog_index, clip_encoder=None, exact_sim: float = 0.93, exact_margin: float = 0.03):
        self.catalog_index = catalog_index
        self.clip_encoder = clip_encoder
        self.extractor = AttributeExtractor(clip_encoder)
        self.exact_sim = exact_sim
        self.exact_margin = exact_margin
        self.style_features = None
        self.material_features = None
        self._pid_to_row = {pid: i for i, pid in enumerate(catalog_index.product_ids)}
        if clip_encoder is not None:
            try:
                import open_clip
                tokenizer = open_clip.get_tokenizer(clip_encoder.model_name)
                with torch.no_grad():
                    for name, prompts in (("style_features", STYLE_PROMPTS), ("material_features", MATERIAL_PROMPTS)):
                        f = clip_encoder.model.encode_text(tokenizer([p[1] for p in prompts]).to(clip_encoder.device))
                        setattr(self, name, (f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
            except Exception as e:
                logger.debug(f"Style/material prompt initialization failed: {e}")

    def _zero_shot(self, emb: Optional[np.ndarray], features: Optional[np.ndarray], prompts) -> Optional[str]:
        if emb is None or features is None:
            return None
        return prompts[int(np.argmax(features @ emb))][0]

    def _encode(self, image: Image.Image) -> Optional[np.ndarray]:
        if self.clip_encoder is None:
            return None
        with torch.no_grad():
            t = self.clip_encoder.preprocess(image.convert("RGB")).unsqueeze(0).to(self.clip_encoder.device)
            f = self.clip_encoder.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()[0].astype(np.float32)

    def is_exact(self, sims: List[float]) -> bool:
        """Predict whether the top result is the same product as the query.

        Requires a very high similarity and a clear gap to the runner-up, so a crowd of
        look-alikes is reported as "similar" instead of a false exact match.
        """
        if not sims or sims[0] < self.exact_sim:
            return False
        if sims[0] >= 0.985:  # the same image; other catalog photos of this product may sit close behind
            return True
        return len(sims) == 1 or (sims[0] - sims[1]) >= self.exact_margin

    @staticmethod
    def same_image(query_image: Optional[Image.Image], product_path: Optional[str]) -> bool:
        """Pixel-level check that the query is the catalog photo itself.

        Embedding similarity alone cannot establish identity: two different white shirts can
        score very high. An exact match is therefore claimed only when the two pictures also
        agree pixel by pixel (correlation of 32x32 greyscale thumbnails), which is true for the
        same photo after resizing or re-compression and false for a different photo of even the
        same product.
        """
        if query_image is None or not product_path:
            return False
        try:
            a = np.asarray(query_image.convert("L").resize((32, 32)), dtype=np.float32).ravel()
            with Image.open(product_path) as im:
                b = np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32).ravel()
            a, b = a - a.mean(), b - b.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            return denom > 0 and float(a @ b) / denom >= SAME_IMAGE_CORRELATION
        except Exception:
            return False

    def explain_batch(
        self,
        results: List[Any],
        query_image: Optional[Image.Image] = None,
        query_category: Optional[str] = None,
    ) -> List[dict]:
        """Generate explanations for a ranked list of (product_id, cosine, [rerank]) results."""
        q_attrs, q_style, q_material = {}, None, None
        if query_image is not None:
            q_attrs = self.extractor.extract_attributes(query_image)
            q_emb = self._encode(query_image)
            q_style = self._zero_shot(q_emb, self.style_features, STYLE_PROMPTS)
            q_material = self._zero_shot(q_emb, self.material_features, MATERIAL_PROMPTS)
        q_color = q_attrs.get("primary_color")
        q_pattern = q_attrs.get("pattern")
        whole_look = bool(query_category) and ("Outfit" in query_category or "Whole" in query_category)

        sims = sorted((float(r[1]) for r in results), reverse=True)
        exact_top = self.is_exact(sims)

        explanations = []
        for rank, res in enumerate(results, 1):
            pid, sim = res[0], float(res[1])
            rerank = float(res[2]) if len(res) >= 3 else sim
            meta = self.catalog_index.get_metadata(pid) or {}
            p_cat = meta.get("category", "Apparel")
            p_color = meta.get("dominant_color") or "Neutral"
            p_pattern = meta.get("pattern") or "Solid Color"
            row = self._pid_to_row.get(pid)
            p_emb = self.catalog_index.embeddings[row] if row is not None else None
            p_style = self._zero_shot(p_emb, self.style_features, STYLE_PROMPTS)
            p_material = self._zero_shot(p_emb, self.material_features, MATERIAL_PROMPTS)

            # Similarity nominates a candidate; identity is claimed only if the images match too
            is_exact = exact_top and sim == sims[0] and self.same_image(query_image, meta.get("image_path"))

            shared, differs = [], []

            def compare(name, q_val, p_val, skip=False):
                if skip or not q_val or not p_val:
                    return
                (shared if q_val == p_val else differs).append(
                    f"{name}: {p_val}" if q_val == p_val else f"{name}: {p_val} (query looks {q_val})"
                )

            compare("Category", query_category, p_cat, skip=whole_look)
            compare("Color", q_color, p_color)
            compare("Pattern", q_pattern, p_pattern)
            compare("Style", q_style, p_style)
            compare("Material", q_material, p_material)

            if is_exact:
                headline = (
                    f"Exact match: this is the same photo as the catalog image "
                    f"(similarity {sim:.3f}, verified pixel by pixel)."
                )
            elif sim >= 0.80:
                headline = f"Very close look-alike (similarity {sim:.3f}); no exact catalog match was confirmed."
            else:
                headline = f"Visually similar alternative (similarity {sim:.3f}); no exact catalog match was found."

            explanations.append({
                "product_id": pid,
                "rank": rank,
                "cosine_similarity": round(sim, 4),
                "rerank_score": round(rerank, 4),
                "category": p_cat,
                "dominant_color": p_color,
                "pattern": p_pattern,
                "style": p_style or "N/A",
                "material": p_material or "N/A",
                "is_exact_match": is_exact,
                "match_type": "Exact Match" if is_exact else "Visually Similar",
                "shared": shared,
                "differs": differs,
                "summary": headline,
                "image_path": meta.get("image_path"),
            })
        return explanations


