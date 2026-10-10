import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProductCatalogLoader:
    """Loads and validates the product catalog from JSONL."""

    def __init__(self, catalog_file: Path):
        self.catalog_file = catalog_file
        self.products = []
        self.unique_product_ids = set()
        self.product_counts = defaultdict(int)

    def load(self, limit: int = None) -> List[str]:
        """Load product IDs from catalog.

        Args:
            limit: Max number of records to load (None = all)

        Returns:
            List of product IDs (may contain duplicates)
        """
        self.products = []
        self.unique_product_ids = set()
        self.product_counts = defaultdict(int)

        with open(self.catalog_file, "r") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    data = json.loads(line.strip())
                    prod_id = data.get("product")
                    if prod_id:
                        self.products.append(prod_id)
                        self.unique_product_ids.add(prod_id)
                        self.product_counts[prod_id] += 1
                except Exception as e:
                    logger.warning(f"Error parsing line {i}: {e}")

        logger.info(f"Loaded {len(self.products)} product records")
        logger.info(f"Unique product IDs: {len(self.unique_product_ids)}")
        logger.info(f"Duplicate entries: {len(self.products) - len(self.unique_product_ids)}")

        return self.products

    def get_unique_products(self) -> Set[str]:
        """Get set of unique product IDs."""
        return self.unique_product_ids

    def get_duplicate_count(self, product_id: str) -> int:
        """Get number of times a product appears."""
        return self.product_counts.get(product_id, 0)


class ValidationLoader:
    """Loads and processes validation data from JSONL."""

    def __init__(self, validation_file: Path):
        self.validation_file = validation_file
        self.validation_pairs = []
        self.scenes = set()
        self.products_in_validation = set()
        self.scene_to_products = defaultdict(list)

    def load(self) -> List[Dict]:
        """Load validation pairs.

        Returns:
            List of {"product": id, "scene": id} dicts
        """
        self.validation_pairs = []
        self.scenes = set()
        self.products_in_validation = set()
        self.scene_to_products = defaultdict(list)

        with open(self.validation_file, "r") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    scene = data.get("scene")
                    prod = data.get("product")
                    if scene and prod:
                        self.validation_pairs.append({"scene": scene, "product": prod})
                        self.scenes.add(scene)
                        self.products_in_validation.add(prod)
                        self.scene_to_products[scene].append(prod)
                except Exception as e:
                    logger.warning(f"Error parsing validation line {i}: {e}")

        logger.info(f"Loaded {len(self.validation_pairs)} validation pairs")
        logger.info(f"Unique scenes: {len(self.scenes)}")
        logger.info(f"Unique products: {len(self.products_in_validation)}")

        return self.validation_pairs

    def get_scene_products(self, scene_id: str) -> List[str]:
        """Get list of ground-truth products for a scene."""
        return self.scene_to_products.get(scene_id, [])

    def get_all_scenes(self) -> Set[str]:
        """Get set of all scenes."""
        return self.scenes

    def get_all_products(self) -> Set[str]:
        """Get set of all products in validation set."""
        return self.products_in_validation

    def get_validation_pairs(self) -> List[Dict]:
        """Get all validation pairs."""
        return self.validation_pairs
