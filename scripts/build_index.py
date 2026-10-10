#!/usr/bin/env python3
"""Build the product catalog index, embeddings, and FAISS vector index."""

import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PRODUCT_CATALOG_FILE, VALIDATION_FILE, EMBEDDINGS_DIR, IMAGES_DIR,
    PRODUCTS_METADATA_FILE, PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_INDEX_FILE,
    MODEL_NAME, PRETRAINED, DEVICE, EMBEDDING_BATCH_SIZE, DOWNLOAD_MAX_WORKERS
)
from src.data_loader import ProductCatalogLoader, ValidationLoader
from src.image_downloader import ImageDownloader
from src.image_encoder import CLIPImageEncoder
from src.catalog_index import ProductIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build product catalog index")
    parser.add_argument("--limit", type=int, default=1500, help="Max unique catalog products to index (default: 1500; use 0 for all 28k)")
    parser.add_argument("--smoke-test", action="store_true", help="Run in tiny smoke test mode (50 products)")
    parser.add_argument("--download-scenes", action="store_true", default=True, help="Download validation scenes for evaluation")
    parser.add_argument("--workers", type=int, default=DOWNLOAD_MAX_WORKERS, help="Parallel download threads")
    return parser.parse_args()


def main():
    args = parse_args()
    limit = 50 if args.smoke_test else (None if args.limit == 0 else args.limit)

    logger.info("=" * 60)
    logger.info(f"BUILDING PRODUCT CATALOG INDEX (Mode: {'SMOKE TEST' if args.smoke_test else f'LIMIT={limit}'})")
    logger.info("=" * 60)

    # 1. Load catalog
    logger.info("1. Ingesting product catalog...")
    catalog_loader = ProductCatalogLoader(PRODUCT_CATALOG_FILE)
    catalog_loader.load()
    unique_catalog_ids = list(catalog_loader.get_unique_products())
    logger.info(f"Total catalog records: {len(catalog_loader.products)}")
    logger.info(f"Unique product IDs: {len(unique_catalog_ids)}")

    # 2. Load validation set to ensure ground-truth products are included
    logger.info("2. Ingesting validation relationships...")
    val_loader = ValidationLoader(VALIDATION_FILE)
    val_loader.load()
    val_products = val_loader.get_all_products()
    val_scenes = list(val_loader.get_all_scenes())

    # Priority set: all validation products that exist in the catalog
    val_in_catalog = [p for p in val_products if p in catalog_loader.get_unique_products()]
    logger.info(f"Validation products in catalog: {len(val_in_catalog)}")

    # Combine validation products with catalog products to form target set
    target_products = []
    seen = set()
    # Always include all validation ground truth items first
    for pid in val_in_catalog:
        target_products.append(pid)
        seen.add(pid)

    # Add remaining catalog items up to limit
    for pid in unique_catalog_ids:
        if pid not in seen:
            target_products.append(pid)
            seen.add(pid)
            if limit and len(target_products) >= limit:
                break

    logger.info(f"Target products to download and index: {len(target_products)}")

    # 3. Download images concurrently
    logger.info("3. Downloading and validating product images...")
    downloader = ImageDownloader(IMAGES_DIR, max_workers=args.workers)
    product_image_paths = downloader.download_images_parallel(target_products, max_workers=args.workers)

    # Download validation scene images
    if args.download_scenes:
        logger.info(f"4. Downloading {len(val_scenes)} validation scene images for benchmark...")
        scene_paths = downloader.download_images_parallel(val_scenes, max_workers=args.workers)
        logger.info(f"Accessible scene images: {len(scene_paths)} / {len(val_scenes)}")

    downloader.save_stats()
    download_stats = downloader.get_stats()
    logger.info(f"Download statistics: {download_stats}")

    # 4. Filter products that have valid local images
    valid_products = []
    valid_paths = []
    for pid in target_products:
        path = product_image_paths.get(pid)
        if path and Path(path).exists() and Path(path).stat().st_size > 0:
            valid_products.append(pid)
            valid_paths.append(Path(path))

    logger.info(f"Valid product images ready for embedding: {len(valid_products)}")

    # 5. Encode images with CLIP in batches
    logger.info(f"5. Encoding images with OpenCLIP ({MODEL_NAME}) on {DEVICE}...")
    encoder = CLIPImageEncoder(model_name=MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)

    product_embeddings = {}
    product_metadata = {}

    total_images = len(valid_paths)
    for i in range(0, total_images, EMBEDDING_BATCH_SIZE):
        batch_paths = valid_paths[i : i + EMBEDDING_BATCH_SIZE]
        batch_pids = valid_products[i : i + EMBEDDING_BATCH_SIZE]

        if i % (EMBEDDING_BATCH_SIZE * 5) == 0:
            logger.info(f"  Encoding progress: {i}/{total_images} ({(i/total_images):.1%})")

        batch_result = encoder.encode_batch(batch_paths, batch_size=len(batch_paths))

        for pid, path in zip(batch_pids, batch_paths):
            emb = batch_result.get(path)
            if emb is not None:
                product_embeddings[pid] = emb
                product_metadata[pid] = {
                    "product_id": pid,
                    "image_path": str(path),
                    "catalog_frequency": catalog_loader.get_duplicate_count(pid),
                }

    logger.info(f"Successfully generated visual embeddings: {len(product_embeddings)}")

    # 6. Build FAISS Index and persist
    logger.info("6. Building FAISS Inner Product index and persisting artifacts...")
    index = ProductIndex(PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_METADATA_FILE, PRODUCTS_INDEX_FILE)
    num_indexed = index.build_index(product_embeddings, product_metadata)
    index.save()

    logger.info("=" * 60)
    logger.info("BUILD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total catalog records:         {len(catalog_loader.products)}")
    logger.info(f"Unique catalog IDs:            {len(unique_catalog_ids)}")
    logger.info(f"Target products processed:     {len(target_products)}")
    logger.info(f"Valid images downloaded:       {len(valid_products)}")
    logger.info(f"Embeddings generated:          {len(product_embeddings)}")
    logger.info(f"FAISS index size:              {num_indexed} products")
    logger.info(f"Embeddings saved to:           {PRODUCTS_EMBEDDINGS_FILE}")
    logger.info(f"Metadata saved to:             {PRODUCTS_METADATA_FILE}")
    logger.info(f"FAISS index saved to:          {PRODUCTS_INDEX_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
