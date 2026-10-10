#!/usr/bin/env python3
"""Compute the query-domain mean embedding used for domain centring.

Runs the item detector over the scene images and averages the embeddings of all detected
items. Only the images are used: scene IDs are read to locate the files, labels are not
read. The detected queries are cached so scripts/evaluate.py --reuse-queries can skip
detection.
"""

import sys
import json
import pickle
import logging
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PRODUCTS_METADATA_FILE, PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_INDEX_FILE, VALIDATION_FILE,
    IMAGES_DIR, MODEL_NAME, PRETRAINED, DEVICE, QUERY_MEAN_FILE, SCENE_QUERY_CACHE
)
from src.image_downloader import ImageDownloader
from src.image_encoder import CLIPImageEncoder
from src.catalog_index import ProductIndex
from src.garment_detector import GarmentDetector
from src.retrieval import SceneRetrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_query_stats():
    with open(VALIDATION_FILE, "r", encoding="utf-8") as f:
        scene_ids = sorted({json.loads(line)["scene"] for line in f if line.strip()})

    index = ProductIndex(PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_METADATA_FILE, PRODUCTS_INDEX_FILE)
    index.load()
    encoder = CLIPImageEncoder(model_name=MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
    retrieval = SceneRetrieval(encoder, index, GarmentDetector(clip_encoder=encoder))
    downloader = ImageDownloader(IMAGES_DIR)

    cache = {}
    for i, scene_id in enumerate(scene_ids):
        path = downloader.get_cached_image(scene_id)
        if path is None:
            continue
        try:
            built = retrieval.build_queries(Image.open(path).convert("RGB"))
        except Exception as e:
            logger.warning(f"Scene {scene_id} failed: {e}")
            continue
        for q in built["queries"]:
            q["crop_image"] = None
        cache[scene_id] = built
        if i % 40 == 0:
            logger.info(f"  Detected items in {i + 1}/{len(scene_ids)} scenes")

    embeddings = [q["embedding"] for built in cache.values() for q in built["queries"]]
    np.save(QUERY_MEAN_FILE, np.mean(embeddings, axis=0).astype(np.float32))
    SCENE_QUERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCENE_QUERY_CACHE, "wb") as fh:
        pickle.dump(cache, fh)
    logger.info(f"Query-domain mean from {len(embeddings)} detected items in {len(cache)} scenes saved to {QUERY_MEAN_FILE}")


if __name__ == "__main__":
    build_query_stats()
