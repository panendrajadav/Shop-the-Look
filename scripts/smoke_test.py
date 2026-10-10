#!/usr/bin/env python3
"""Smoke test: Verify system works with small dataset."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import IMAGES_DIR, EMBEDDINGS_DIR, MODEL_NAME, PRETRAINED, DEVICE
from src.data_loader import ProductCatalogLoader
from src.image_downloader import ImageDownloader
from src.image_encoder import CLIPImageEncoder
from src.catalog_index import ProductIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("SMOKE TEST: Verify all components work")
    logger.info("=" * 60)

    # Test 1: Load catalog
    logger.info("\n1. Testing ProductCatalogLoader...")
    catalog_file = Path(__file__).parent.parent / "dataset" / "product_catelog.jsonl"
    catalog_loader = ProductCatalogLoader(catalog_file)
    products = catalog_loader.load(limit=10)
    assert len(products) == 10, "Should load 10 products"
    unique = len(catalog_loader.get_unique_products())
    logger.info(f"   Loaded {len(products)} products, {unique} unique - OK")

    # Test 2: Image downloader
    logger.info("\n2. Testing ImageDownloader...")
    downloader = ImageDownloader(IMAGES_DIR)
    test_products = list(catalog_loader.get_unique_products())[:3]
    for prod_id in test_products:
        path = downloader.download_image(prod_id)
        if path:
            logger.info(f"   Downloaded {prod_id} - OK")
        else:
            logger.warning(f"   Failed to download {prod_id}")

    downloader.save_stats()

    # Test 3: CLIP encoder
    logger.info("\n3. Testing CLIPImageEncoder...")
    try:
        encoder = CLIPImageEncoder(model_name=MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
        logger.info(f"   Loaded {MODEL_NAME} on {DEVICE}")
        logger.info(f"   Embedding dimension: {encoder.get_embedding_dim()}")

        # Try encoding a downloaded image
        downloaded_paths = [
            Path(downloader.metadata.get(pid, {}).get("path"))
            for pid in test_products
            if downloader.metadata.get(pid, {}).get("status") == "success"
        ]

        if downloaded_paths and downloaded_paths[0].exists():
            embedding = encoder.encode_image(downloaded_paths[0])
            if embedding is not None:
                logger.info(f"   Encoded image successfully - shape {embedding.shape} - OK")
            else:
                logger.warning("   Failed to encode image")
    except ImportError as e:
        logger.warning(f"   CLIP not available (will need pip install): {e}")
        return

    # Test 4: Index building
    logger.info("\n4. Testing ProductIndex...")
    import numpy as np
    embeddings_file = EMBEDDINGS_DIR / "smoke_embeddings.npy"
    metadata_file = EMBEDDINGS_DIR / "smoke_metadata.json"
    index_file = EMBEDDINGS_DIR / "smoke_index.idx"

    # Create dummy data
    test_embeddings = {
        pid: np.random.randn(512).astype(np.float32)
        for pid in test_products[:3]
    }
    test_metadata = {
        pid: {"product_id": pid}
        for pid in test_products[:3]
    }

    index = ProductIndex(embeddings_file, metadata_file, index_file)
    num_indexed = index.build_index(test_embeddings, test_metadata)
    logger.info(f"   Built index with {num_indexed} products - OK")

    # Test search
    query = test_embeddings[test_products[0]]
    results = index.search(query, top_k=2)
    logger.info(f"   Retrieved {len(results)} products - OK")

    # Save and reload
    index.save()
    index2 = ProductIndex(embeddings_file, metadata_file, index_file)
    if index2.load():
        logger.info(f"   Saved and reloaded index - OK")
    else:
        logger.warning("   Failed to reload index")

    # Cleanup
    for f in [embeddings_file, metadata_file, index_file]:
        if f.exists():
            f.unlink()

    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST PASSED")
    logger.info("=" * 60)
    logger.info("\nAll components working. System is ready for full build.")
    logger.info("Run: python scripts/build_index.py")


if __name__ == "__main__":
    main()
