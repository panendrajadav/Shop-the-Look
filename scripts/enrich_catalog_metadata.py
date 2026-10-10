import sys
from pathlib import Path
import json
import colorsys
import numpy as np
import torch
from PIL import Image
import open_clip

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

CATEGORY_TAXONOMY = {
    "Jacket / Outerwear": [
        "a photo of a jacket", "an outerwear coat", "a blazer", "a winter coat",
        "a leather jacket", "a denim jacket", "a cardigan sweater", "a hoodie jacket"
    ],
    "Top / T-shirt": [
        "a photo of a t-shirt", "a graphic t-shirt", "a casual shirt", "a button-up shirt",
        "a blouse", "a tank top", "a crop top", "a long sleeve shirt"
    ],
    "Pants / Trousers": [
        "a photo of pants", "denim jeans", "trousers", "slacks", "chinos", "sweatpants", "shorts"
    ],
    "Dress": [
        "a photo of a dress", "an evening gown", "a summer dress", "a cocktail dress", "a mini dress", "a midi dress"
    ],
    "Skirt": [
        "a photo of a skirt", "a mini skirt", "a pleated skirt", "a pencil skirt", "a midi skirt"
    ],
    "Footwear": [
        "a photo of shoes", "sneakers", "high heels", "leather boots", "ankle boots", "sandals", "loafers"
    ],
    "Bag / Handbag": [
        "a photo of a handbag", "a purse", "a tote bag", "a backpack", "a shoulder bag", "a clutch"
    ],
}

PATTERN_TAXONOMY = {
    "Solid Color": "a photo of solid single color plain fabric",
    "Graphic / Printed": "a photo of graphic printed or logo fabric",
    "Striped": "a photo of striped fabric pattern",
    "Floral / Patterned": "a photo of floral or patterned decorative fabric",
    "Plaid / Checkered": "a photo of plaid or checkered pattern fabric",
}


def extract_dominant_color_from_image(im: Image.Image) -> str:
    """Robust foreground color extraction with border background removal and HSV quantization."""
    im_small = im.convert("RGB").resize((100, 100))
    arr = np.array(im_small)
    h, w, _ = arr.shape

    # Sample border pixels for background estimate
    borders = np.concatenate([
        arr[:5, :].reshape(-1, 3),
        arr[-5:, :].reshape(-1, 3),
        arr[:, :5].reshape(-1, 3),
        arr[:, -5:].reshape(-1, 3),
    ], axis=0)
    bg_color = np.median(borders, axis=0)

    pixels = arr.reshape(-1, 3).astype(float)
    dist = np.linalg.norm(pixels - bg_color, axis=1)

    # Background mask: close to border color OR bright neutral white
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

    # Classify color in HSV
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
    print("1. Loading existing metadata and embeddings...")
    with open(config.PRODUCTS_METADATA_FILE, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    product_ids = meta_data["product_ids"]
    metadata_dict = meta_data["metadata"]
    embeddings = np.load(config.PRODUCTS_EMBEDDINGS_FILE)

    print(f"Loaded {len(product_ids)} products with embedding shape {embeddings.shape}")

    print("2. Computing category text embeddings using OpenCLIP...")
    model, _, _ = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.PRETRAINED)
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)

    cat_names = list(CATEGORY_TAXONOMY.keys())
    cat_vectors = []
    for cat in cat_names:
        toks = tokenizer(CATEGORY_TAXONOMY[cat])
        with torch.no_grad():
            tf = model.encode_text(toks)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            mean_f = tf.mean(dim=0)
            mean_f = mean_f / mean_f.norm(dim=-1, keepdim=True)
            cat_vectors.append(mean_f.numpy())
    cat_matrix = np.stack(cat_vectors)  # (7, 512)

    # Pattern matrix
    pat_names = list(PATTERN_TAXONOMY.keys())
    pat_vectors = []
    for p in pat_names:
        toks = tokenizer([PATTERN_TAXONOMY[p]])
        with torch.no_grad():
            pf = model.encode_text(toks)
            pf = pf / pf.norm(dim=-1, keepdim=True)
            pat_vectors.append(pf.numpy()[0])
    pat_matrix = np.stack(pat_vectors)  # (5, 512)

    print("3. Classifying categories, patterns, and extracting dominant colors...")
    cat_sims = embeddings @ cat_matrix.T  # (1000, 7)
    cat_indices = np.argmax(cat_sims, axis=1)

    # Softmax for category confidence
    exp_sims = np.exp(cat_sims * 15.0)
    cat_probs = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)

    pat_sims = embeddings @ pat_matrix.T  # (1000, 5)
    pat_indices = np.argmax(pat_sims, axis=1)

    category_counts = {c: 0 for c in cat_names}
    color_counts = {}

    for i, pid in enumerate(product_ids):
        best_cat = cat_names[cat_indices[i]]
        conf = float(cat_probs[i, cat_indices[i]])
        category_counts[best_cat] += 1

        best_pat = pat_names[pat_indices[i]]

        # Image color extraction
        item_meta = metadata_dict.get(pid, {})
        img_path = Path(item_meta.get("image_path", ""))

        if img_path.exists():
            try:
                with Image.open(img_path) as im:
                    dom_color = extract_dominant_color_from_image(im)
            except Exception:
                dom_color = "Neutral"
        else:
            dom_color = "Neutral"

        color_counts[dom_color] = color_counts.get(dom_color, 0) + 1

        # Enrich metadata record
        item_meta["category"] = best_cat
        item_meta["category_confidence"] = round(conf, 4)
        item_meta["pattern"] = best_pat
        item_meta["dominant_color"] = dom_color
        metadata_dict[pid] = item_meta

    print("\nCategory Distribution:")
    for c, count in category_counts.items():
        print(f"  {c:20s}: {count:4d}")

    print("\nDominant Color Distribution:")
    for col, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {col:20s}: {count:4d}")

    print("\n4. Saving enriched metadata to products_metadata.json...")
    with open(config.PRODUCTS_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    print("Enrichment complete! Metadata successfully saved.")


if __name__ == "__main__":
    enrich_catalog()
