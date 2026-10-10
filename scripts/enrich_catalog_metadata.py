#!/usr/bin/env python3
"""Estimate item name, category, pattern and dominant color for every indexed product.

The catalog file holds IDs only, so these attributes come from the product image:
item name and category by zero-shot CLIP over the item vocabulary shared with the scene
detector, pattern by zero-shot CLIP, and color from the image pixels.
"""

import sys
from pathlib import Path
import json
import colorsys
import numpy as np
import torch
from PIL import Image
import open_clip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.garment_detector import ITEM_NAMES, PATTERN_PROMPTS as PATTERN_TAXONOMY
from src.color_features import color_histogram, product_color_box

def extract_dominant_color_from_image(im: Image.Image) -> str:
    """Robust foreground color extraction with border background removal and HSV quantization."""
    im_small = im.convert("RGB").resize((100, 100))
    arr = np.array(im_small)
    h, w, _ = arr.shape

    borders = np.concatenate([
        arr[:5, :].reshape(-1, 3),
        arr[-5:, :].reshape(-1, 3),
        arr[:, :5].reshape(-1, 3),
        arr[:, -5:].reshape(-1, 3),
    ], axis=0)
    bg_color = np.median(borders, axis=0)

    pixels = arr.reshape(-1, 3).astype(float)
    dist = np.linalg.norm(pixels - bg_color, axis=1)

    is_bg = (dist < 38) | (
        (pixels[:, 0] > 218) & (pixels[:, 1] > 218) & (pixels[:, 2] > 218) & (np.ptp(pixels, axis=1) < 22)
    )

    if is_bg.sum() > 0.95 * len(pixels):
        fg_pixels = arr[25:75, 25:75].reshape(-1, 3)
    else:
        fg_pixels = arr.reshape(-1, 3)[~is_bg]

    if len(fg_pixels) == 0:
        return "Neutral"

    med_rgb = np.median(fg_pixels, axis=0)
    r, g, b = med_rgb
    h_norm, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h_deg = h_norm * 360.0

    if v < 0.22 or (v < 0.35 and s < 0.22):
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


def enrich_catalog():
    with open(config.PRODUCTS_METADATA_FILE, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    product_ids = meta_data["product_ids"]
    metadata_dict = meta_data["metadata"]
    embeddings = np.load(config.PRODUCTS_EMBEDDINGS_FILE)
    print(f"Enriching {len(product_ids)} products")

    model, _, _ = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.PRETRAINED)
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)

    def encode(prompts):
        with torch.no_grad():
            tf = model.encode_text(tokenizer(prompts))
            return (tf / tf.norm(dim=-1, keepdim=True)).numpy()

    items = [(cat, name, prompt) for cat, entries in ITEM_NAMES.items() for name, prompt in entries]
    item_sims = embeddings @ encode([f"a photo of {p}" for _, _, p in items]).T
    item_idx = item_sims.argmax(axis=1)
    probs = np.exp((item_sims - item_sims.max(axis=1, keepdims=True)) * 100.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    pat_names = list(PATTERN_TAXONOMY.keys())
    pat_idx = (embeddings @ encode(list(PATTERN_TAXONOMY.values())).T).argmax(axis=1)

    cat_names = list(ITEM_NAMES.keys())
    item_cat = np.array([cat_names.index(c) for c, _, _ in items])
    cat_probs = np.stack([probs[:, item_cat == k].sum(axis=1) for k in range(len(cat_names))], axis=1)
    cat_order = np.argsort(-cat_probs, axis=1)

    color_hists = np.zeros((len(product_ids), 28), dtype=np.float32)
    category_counts = {}
    for i, pid in enumerate(product_ids):
        cat, name, _ = items[item_idx[i]]
        item_meta = metadata_dict.get(pid, {})
        item_meta["category"] = cat
        item_meta["item_name"] = name
        item_meta["category_confidence"] = round(float(cat_probs[i, cat_names.index(cat)]), 4)
        alt = cat_names[int(cat_order[i, 0])] if cat_names[int(cat_order[i, 0])] != cat else cat_names[int(cat_order[i, 1])]
        item_meta["category_alt"] = alt
        item_meta["category_alt_confidence"] = round(float(cat_probs[i, cat_names.index(alt)]), 4)
        try:
            with Image.open(item_meta.get("image_path", "")) as im:
                im = im.convert("RGB")
                color_hists[i] = color_histogram(im, box=product_color_box(cat), remove_background=True)
                if not item_meta.get("dominant_color"):
                    item_meta["dominant_color"] = extract_dominant_color_from_image(im)
        except Exception:
            pass
        item_meta["pattern"] = pat_names[pat_idx[i]]
        item_meta.setdefault("dominant_color", "Neutral")
        metadata_dict[pid] = item_meta
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("Category distribution:")
    for c, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:20s}: {count:5d}")

    with open(config.PRODUCTS_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)
    np.save(config.PRODUCTS_COLOR_FILE, color_hists)
    print("Metadata and color histograms saved.")


if __name__ == "__main__":
    enrich_catalog()
