#!/usr/bin/env python3
"""Build the product catalog index, embeddings, and FAISS vector index.

Only products listed in the catalog file are indexed. Validation labels are never
read here, so nothing from validation.jsonl can leak into the searchable index.
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PRODUCT_CATALOG_FILE, VALIDATION_FILE, IMAGES_DIR,
    PRODUCTS_METADATA_FILE, PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_INDEX_FILE,
    MODEL_NAME, PRETRAINED, DEVICE, EMBEDDING_BATCH_SIZE, DOWNLOAD_MAX_WORKERS
)
from src.data_loader import ProductCatalogLoader, ValidationLoader
from src.image_downloader import ImageDownloader
from src.image_encoder import CLIPImageEncoder
from src.catalog_index import ProductIndex
from enrich_catalog_metadata import enrich_catalog
from build_product_views import build_product_views
from build_query_stats import build_query_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build product catalog index")
    parser.add_argument("--limit", type=int, default=0, help="Max unique catalog products to index (default 0 = all)")
    parser.add_argument("--cached-only", action="store_true", help="Skip downloading; index only images already in the local cache")
    parser.add_argument("--max-minutes", type=float, default=0, help="Stop encoding after this many minutes and save what is done (0 = no limit)")
    parser.add_argument("--workers", type=int, default=DOWNLOAD_MAX_WORKERS, help="Parallel download threads")
    return parser.parse_args()


def main():
    args = parse_args()

    catalog_loader = ProductCatalogLoader(PRODUCT_CATALOG_FILE)
    catalog_loader.load()
    catalog_ids = sorted(catalog_loader.get_unique_products())
    catalog_set = set(catalog_ids)
    logger.info(f"Catalog records: {len(catalog_loader.products)} | unique product IDs: {len(catalog_ids)}")

    product_embeddings = {}
    product_metadata = {}
    if PRODUCTS_EMBEDDINGS_FILE.exists() and PRODUCTS_METADATA_FILE.exists():
        try:
            existing_embs = np.load(PRODUCTS_EMBEDDINGS_FILE)
            with open(PRODUCTS_METADATA_FILE, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
            for pid, emb in zip(meta_json.get("product_ids", []), existing_embs):
                if pid in catalog_set:
                    product_embeddings[pid] = emb
                    product_metadata[pid] = meta_json.get("metadata", {}).get(pid, {"product_id": pid})
            logger.info(f"Reusing {len(product_embeddings)} existing catalog embeddings.")
        except Exception as e:
            logger.warning(f"Could not load previous embeddings: {e}")

    target_products = [p for p in catalog_ids if p in product_embeddings] + \
                      [p for p in catalog_ids if p not in product_embeddings]
    if args.limit:
        target_products = target_products[:args.limit]

    downloader = ImageDownloader(IMAGES_DIR, max_workers=args.workers)
    if args.cached_only:
        image_paths = {}
        for pid in target_products:
            path = downloader.get_cached_image(pid)
            if path:
                image_paths[pid] = path
    else:
        image_paths = downloader.download_images_parallel(target_products, max_workers=args.workers)
        val_loader = ValidationLoader(VALIDATION_FILE)
        val_loader.load()
        scene_paths = downloader.download_images_parallel(sorted(val_loader.get_all_scenes()), max_workers=args.workers)
        logger.info(f"Scene images available (queries only, never indexed): {len(scene_paths)}")
        logger.info(f"Download statistics: {downloader.get_stats()}")
    logger.info(f"Catalog images available locally: {len(image_paths)} / {len(target_products)}")

    to_encode = [(pid, Path(image_paths[pid])) for pid in target_products
                 if pid in image_paths and pid not in product_embeddings]
    logger.info(f"New product images to encode: {len(to_encode)} with OpenCLIP {MODEL_NAME} on {DEVICE}")
    encoder = CLIPImageEncoder(model_name=MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)

    def _load(item):
        try:
            return encoder.preprocess(Image.open(item[1]).convert("RGB"))
        except Exception:
            return None

    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0, len(to_encode), EMBEDDING_BATCH_SIZE):
            if args.max_minutes and (time.time() - start) > args.max_minutes * 60:
                logger.info(f"Time budget reached after {i} images; saving partial index.")
                break
            batch = to_encode[i:i + EMBEDDING_BATCH_SIZE]
            tensors = list(pool.map(_load, batch))
            keep = [(item, t) for item, t in zip(batch, tensors) if t is not None]
            if not keep:
                continue
            with torch.no_grad():
                feats = encoder.model.encode_image(torch.stack([t for _, t in keep]).to(encoder.device))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            for (pid, path), emb in zip([k[0] for k in keep], feats.cpu().numpy().astype(np.float32)):
                product_embeddings[pid] = emb
                product_metadata[pid] = {
                    "product_id": pid,
                    "image_path": str(path),
                    "catalog_frequency": catalog_loader.get_duplicate_count(pid),
                }
            if (i // EMBEDDING_BATCH_SIZE) % 20 == 0:
                rate = (i + len(batch)) / max(1e-6, time.time() - start)
                logger.info(f"  Encoded {i + len(batch)}/{len(to_encode)} ({rate:.1f} img/s)")

    index = ProductIndex(PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_METADATA_FILE, PRODUCTS_INDEX_FILE)
    num_indexed = index.build_index(product_embeddings, product_metadata)
    index.save()

    enrich_catalog()

    build_product_views()

    build_query_stats()

    assert set(index.product_ids) <= catalog_set, "Index contains a non-catalog product"
    logger.info("=" * 60)
    logger.info(f"Unique catalog IDs:        {len(catalog_ids)}")
    logger.info(f"Images available locally:  {len(image_paths)}")
    logger.info(f"FAISS index size:          {num_indexed} products")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
