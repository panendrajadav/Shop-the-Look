#!/usr/bin/env python3
"""Run quantitative evaluation on the validation dataset."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PRODUCTS_METADATA_FILE, PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_INDEX_FILE,
    VALIDATION_FILE, IMAGES_DIR, MODEL_NAME, PRETRAINED, DEVICE,
    VALIDATION_RECALL_KS, EVAL_RESULTS_JSON, EVAL_RESULTS_CSV
)
from src.data_loader import ValidationLoader
from src.image_downloader import ImageDownloader
from src.image_encoder import CLIPImageEncoder
from src.catalog_index import ProductIndex
from src.evaluation import EvaluationPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("EVALUATING VISUAL RETRIEVAL PERFORMANCE")
    logger.info("=" * 60)

    # 1. Load catalog index
    logger.info("1. Loading persisted catalog FAISS index...")
    index = ProductIndex(PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_METADATA_FILE, PRODUCTS_INDEX_FILE)
    if not index.load():
        logger.error("Failed to load catalog index. Please run scripts/build_index.py first.")
        sys.exit(1)

    logger.info(f"Loaded FAISS index containing {index.get_size()} catalog products.")

    # 2. Load CLIP encoder
    logger.info(f"2. Loading OpenCLIP ({MODEL_NAME}) on {DEVICE}...")
    encoder = CLIPImageEncoder(model_name=MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)

    # 3. Load image downloader
    logger.info("3. Initializing image downloader for scene images...")
    downloader = ImageDownloader(IMAGES_DIR)

    # 4. Load validation set
    logger.info(f"4. Loading validation data from {VALIDATION_FILE}...")
    val_loader = ValidationLoader(VALIDATION_FILE)
    validation_pairs = val_loader.load()
    scene_to_products = val_loader.scene_to_products

    # 5. Run evaluation
    from src.garment_detector import GarmentDetector
    detector = GarmentDetector(clip_encoder=encoder, use_torchvision_detector=True)

    logger.info(f"5. Running evaluation across validation scenes (Recall@Ks: {VALIDATION_RECALL_KS})...")
    pipeline = EvaluationPipeline(
        image_downloader=downloader,
        image_encoder=encoder,
        catalog_index=index,
        garment_detector=detector,
        recalls_at=VALIDATION_RECALL_KS,
    )

    results = pipeline.evaluate(
        validation_pairs=validation_pairs,
        scene_to_products=scene_to_products,
    )

    # 6. Display summary
    pipeline.print_summary()

    # 7. Save outputs
    pipeline.evaluator.save_results(json_path=EVAL_RESULTS_JSON, csv_path=EVAL_RESULTS_CSV)
    logger.info(f"Evaluation report saved to: {EVAL_RESULTS_JSON}")
    logger.info(f"Per-scene predictions CSV saved to: {EVAL_RESULTS_CSV}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
