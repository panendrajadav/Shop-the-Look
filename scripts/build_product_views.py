#!/usr/bin/env python3
"""Add garment-level views (upper and lower body crops) for catalog garment photos.

Many catalog images show a model wearing the product together with other clothes, so the
whole-image embedding describes the outfit, not the product. For tops, jackets, pants and
skirts this script also embeds an upper-body and a lower-body crop. At query time a scene's top/jacket crop is
compared with product upper-body views and a pants/skirt crop with lower-body views.

Uses catalog images only; validation labels are not read.
"""

import sys
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PRODUCTS_METADATA_FILE, PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_VIEWS_FILE,
    MODEL_NAME, PRETRAINED, DEVICE, EMBEDDING_BATCH_SIZE
)
from src.image_encoder import CLIPImageEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VIEW_CATEGORIES = {"Top / T-shirt", "Jacket / Outerwear", "Pants / Trousers", "Skirt"}
# (region name, fractional box) - same body bands used for scenes without a detected person
VIEW_BOXES = [("upper", (0.08, 0.05, 0.92, 0.58)), ("lower", (0.10, 0.42, 0.90, 0.92))]


def build_product_views():
    embeddings = np.load(PRODUCTS_EMBEDDINGS_FILE)
    with open(PRODUCTS_METADATA_FILE, "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    product_ids, metadata = meta_json["product_ids"], meta_json["metadata"]

    # Garment photos are often taken on a model and framed around the torso or legs, with a
    # second garment in shot. Every product whose estimated category is a top, jacket, pants
    # or skirt therefore gets both an upper and a lower view.
    encoder = CLIPImageEncoder(model_name=MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
    has_views = np.array([metadata[pid].get("category") in VIEW_CATEGORIES for pid in product_ids])
    # Incremental: products that already have views (from an earlier run) are not re-encoded
    old = np.load(PRODUCTS_VIEWS_FILE) if PRODUCTS_VIEWS_FILE.exists() else None
    done = set(old["rows"].tolist()) if old is not None else set()
    rows = np.array([r for r in np.where(has_views)[0] if r not in done], dtype=np.int64)
    logger.info(f"Products needing garment views: {len(rows)} (already done: {len(done)}, catalog: {len(product_ids)})")

    def _views(row):
        try:
            img = Image.open(metadata[product_ids[row]]["image_path"]).convert("RGB")
            w, h = img.size
            return [encoder.preprocess(img.crop((int(a * w), int(b * h), int(c * w), int(d * h))))
                    for _, (a, b, c, d) in VIEW_BOXES]
        except Exception:
            return None

    view_embs, view_rows, view_regions = [], [], []
    if old is not None and len(old["rows"]):
        view_embs.append(old["embeddings"])
        view_rows.extend(old["rows"].tolist())
        view_regions.extend(old["regions"].tolist())
    per_batch = max(1, EMBEDDING_BATCH_SIZE // len(VIEW_BOXES))
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0, len(rows), per_batch):
            batch_rows = rows[i:i + per_batch]
            tensors, owners = [], []
            for row, views in zip(batch_rows, pool.map(_views, batch_rows)):
                if views is None:
                    continue
                for (region, _), tensor in zip(VIEW_BOXES, views):
                    tensors.append(tensor)
                    owners.append((int(row), region))
            if not tensors:
                continue
            with torch.no_grad():
                feats = encoder.model.encode_image(torch.stack(tensors).to(encoder.device))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            view_embs.append(feats.cpu().numpy().astype(np.float32))
            view_rows.extend(o[0] for o in owners)
            view_regions.extend(o[1] for o in owners)
            if (i // per_batch) % 20 == 0:
                done = i + len(batch_rows)
                logger.info(f"  Views for {done}/{len(rows)} products ({done / max(1e-6, time.time() - start):.1f} products/s)")

    np.savez(
        PRODUCTS_VIEWS_FILE,
        embeddings=np.concatenate(view_embs) if view_embs else np.zeros((0, embeddings.shape[1]), np.float32),
        rows=np.array(view_rows, dtype=np.int64),
        regions=np.array(view_regions),
    )
    for row, pid in enumerate(product_ids):
        metadata[pid]["garment_views"] = bool(has_views[row])
    with open(PRODUCTS_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)
    logger.info(f"Saved {len(view_rows)} garment views to {PRODUCTS_VIEWS_FILE}")


if __name__ == "__main__":
    build_product_views()
