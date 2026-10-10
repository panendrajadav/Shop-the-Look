#!/usr/bin/env python3
"""Test that all required modules can be imported."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing imports...")

# Test core modules (no external deps)
try:
    import config
    print("[OK] config")
except ImportError as e:
    print(f"[FAIL] config: {e}")

try:
    from src.data_loader import ProductCatalogLoader, ValidationLoader
    print("[OK] src.data_loader")
except ImportError as e:
    print(f"[FAIL] src.data_loader: {e}")

try:
    from src.image_downloader import ImageDownloader
    print("[OK] src.image_downloader")
except ImportError as e:
    print(f"[FAIL] src.image_downloader: {e}")

try:
    from src.evaluation import RetrievalEvaluator, EvaluationPipeline
    print("[OK] src.evaluation")
except ImportError as e:
    print(f"[FAIL] src.evaluation: {e}")

try:
    from src.explanations import RecommendationExplainer
    print("[OK] src.explanations")
except ImportError as e:
    print(f"[FAIL] src.explanations: {e}")

# Test ML modules (require installation)
try:
    from src.image_encoder import CLIPImageEncoder
    print("[OK] src.image_encoder (CLIP available)")
except ImportError as e:
    print(f"[WARN] src.image_encoder: {e}")

try:
    from src.catalog_index import ProductIndex
    print("[OK] src.catalog_index (FAISS available)")
except ImportError as e:
    print(f"[WARN] src.catalog_index: {e}")

try:
    from src.retrieval import SceneRetrieval
    print("[OK] src.retrieval")
except ImportError as e:
    print(f"[WARN] src.retrieval: {e}")

print("\nImport test complete.")
print("\nTo install all dependencies, run:")
print("  pip install -r requirements.txt")
