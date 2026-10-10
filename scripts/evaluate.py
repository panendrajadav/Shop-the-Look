#!/usr/bin/env python3
"""Run quantitative evaluation on the validation dataset."""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PRODUCTS_METADATA_FILE, PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_INDEX_FILE,
    VALIDATION_FILE, IMAGES_DIR, MODEL_NAME, PRETRAINED, DEVICE,
    VALIDATION_RECALL_KS, EVAL_RESULTS_JSON, EVAL_RESULTS_CSV, RESULTS_DIR
)
from src.data_loader import ValidationLoader
from src.image_downloader import ImageDownloader
from src.image_encoder import CLIPImageEncoder
from src.catalog_index import ProductIndex
from src.evaluation import EvaluationPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on validation.jsonl")
    parser.add_argument("--reuse-queries", action="store_true",
                        help="Reuse cached detections and query embeddings (valid only if detection/encoding code is unchanged)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EVALUATING VISUAL RETRIEVAL PERFORMANCE")
    logger.info("=" * 60)

    logger.info("1. Loading persisted catalog FAISS index...")
    index = ProductIndex(PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_METADATA_FILE, PRODUCTS_INDEX_FILE)
    if not index.load():
        logger.error("Failed to load catalog index. Please run scripts/build_index.py first.")
        sys.exit(1)

    logger.info(f"Loaded FAISS index containing {index.get_size()} catalog products.")

    logger.info(f"2. Loading OpenCLIP ({MODEL_NAME}) on {DEVICE}...")
    encoder = CLIPImageEncoder(model_name=MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)

    logger.info("3. Initializing image downloader for scene images...")
    downloader = ImageDownloader(IMAGES_DIR)

    logger.info(f"4. Loading validation data from {VALIDATION_FILE}...")
    val_loader = ValidationLoader(VALIDATION_FILE)
    validation_pairs = val_loader.load()
    scene_to_products = val_loader.scene_to_products

    from src.garment_detector import GarmentDetector
    detector = GarmentDetector(clip_encoder=encoder, use_torchvision_detector=True)

    logger.info(f"5. Running evaluation across validation scenes (Recall@Ks: {VALIDATION_RECALL_KS})...")
    pipeline = EvaluationPipeline(
        image_downloader=downloader,
        image_encoder=encoder,
        catalog_index=index,
        garment_detector=detector,
        recalls_at=VALIDATION_RECALL_KS,
        query_cache=RESULTS_DIR / "scene_queries.pkl",
        reuse_queries=args.reuse_queries,
    )

    results = pipeline.evaluate(
        validation_pairs=validation_pairs,
        scene_to_products=scene_to_products,
    )

    pipeline.print_summary()

    pipeline.evaluator.save_results(json_path=EVAL_RESULTS_JSON, csv_path=EVAL_RESULTS_CSV)
    pipeline.save_breakdown(RESULTS_DIR / "scene_breakdown.csv")
    logger.info(f"Per-scene breakdown saved to: {RESULTS_DIR / 'scene_breakdown.csv'}")
    logger.info(f"Evaluation report saved to: {EVAL_RESULTS_JSON}")
    logger.info(f"Per-scene predictions CSV saved to: {EVAL_RESULTS_CSV}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
