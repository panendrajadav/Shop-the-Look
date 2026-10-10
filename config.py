import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
# Dataset directory (check data/ first, fallback to dataset/)
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
IMAGES_DIR = ARTIFACTS_DIR / "images"
RESULTS_DIR = ARTIFACTS_DIR / "results"

# Create directories if needed
for dir_path in [ARTIFACTS_DIR, EMBEDDINGS_DIR, IMAGES_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset files
PRODUCT_CATALOG_FILE = DATA_DIR / "product_catelog.jsonl"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"

# Saved indexes
PRODUCTS_METADATA_FILE = EMBEDDINGS_DIR / "products_metadata.json"
PRODUCTS_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "products_embeddings.npy"
PRODUCTS_INDEX_FILE = EMBEDDINGS_DIR / "products_faiss.index"

# Model configuration
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = "cuda" if (os.environ.get("CUDA_AVAILABLE") and os.environ.get("CUDA_AVAILABLE").lower() != "false") else "cpu"

# Embedding configuration
EMBEDDING_DIM = 512
EMBEDDING_BATCH_SIZE = 32

# Image download configuration
IMAGE_TIMEOUT = 8
IMAGE_RETRY_ATTEMPTS = 3
DOWNLOAD_MAX_WORKERS = 16
IMAGE_TARGET_SIZE = (224, 224)
MAX_IMAGES_TO_DOWNLOAD = None  # None means download all
EVAL_RESULTS_JSON = RESULTS_DIR / "evaluation_results.json"
EVAL_RESULTS_CSV = RESULTS_DIR / "per_scene_evaluation.csv"

# Retrieval configuration
DEFAULT_TOP_K = 3
MAX_TOP_K = 20

# Validation configuration
VALIDATION_RECALL_KS = [1, 5, 10]

# Enable debug logging
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
