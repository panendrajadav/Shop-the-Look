#!/usr/bin/env python3
"""Unit tests for JSONL parsing, duplicate handling, invalid IDs, and data loading."""

import sys
import json
import tempfile
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import ProductCatalogLoader, ValidationLoader


def test_product_catalog_loader_duplicates_and_counts():
    """Verify JSONL parsing, duplicate tracking, and record counting."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write('{"product": "prod_A"}\n')
        f.write('{"product": "prod_A"}\n')
        f.write('{"product": "prod_B"}\n')
        f.write('{"product": "prod_C"}\n')
        temp_file = Path(f.name)

    try:
        loader = ProductCatalogLoader(temp_file)
        products = loader.load()

        assert len(products) == 4
        assert len(loader.get_unique_products()) == 3
        assert loader.get_duplicate_count("prod_A") == 2
        assert loader.get_duplicate_count("prod_B") == 1
    finally:
        if temp_file.exists():
            temp_file.unlink()


def test_product_catalog_invalid_lines_and_missing_keys():
    """Verify handling of invalid JSON syntax, empty lines, and missing keys."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write('{"product": "valid_1"}\n')
        f.write('not a valid json string\n')
        f.write('\n')
        f.write('{"other_key": "no_product_field"}\n')
        f.write('{"product": ""}\n')
        f.write('{"product": "valid_2"}\n')
        temp_file = Path(f.name)

    try:
        loader = ProductCatalogLoader(temp_file)
        products = loader.load()

        assert len(products) == 2
        assert set(loader.get_unique_products()) == {"valid_1", "valid_2"}
    finally:
        if temp_file.exists():
            temp_file.unlink()


def test_empty_catalog_file():
    """Verify graceful handling of an empty JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        temp_file = Path(f.name)

    try:
        loader = ProductCatalogLoader(temp_file)
        products = loader.load()
        assert len(products) == 0
        assert len(loader.get_unique_products()) == 0
    finally:
        if temp_file.exists():
            temp_file.unlink()


def test_validation_loader_grouping_and_multiple_products():
    """Verify scene grouping and scenes with multiple ground truth products."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write('{"product": "p1", "scene": "s1"}\n')
        f.write('{"product": "p2", "scene": "s1"}\n')
        f.write('{"product": "p3", "scene": "s2"}\n')
        temp_file = Path(f.name)

    try:
        loader = ValidationLoader(temp_file)
        pairs = loader.load()

        assert len(pairs) == 3
        assert len(loader.get_all_scenes()) == 2
        assert len(loader.get_all_products()) == 3

        # Scene s1 has 2 ground truth products
        s1_prods = loader.get_scene_products("s1")
        assert set(s1_prods) == {"p1", "p2"}

        # Scene s2 has 1 ground truth product
        s2_prods = loader.get_scene_products("s2")
        assert s2_prods == ["p3"]
    finally:
        if temp_file.exists():
            temp_file.unlink()


def test_real_dataset_integrity():
    """Verify that real catalog and validation files exist and match expected schemas."""
    from config import PRODUCT_CATALOG_FILE, VALIDATION_FILE

    if PRODUCT_CATALOG_FILE.exists():
        cat_loader = ProductCatalogLoader(PRODUCT_CATALOG_FILE)
        products = cat_loader.load(limit=200)
        assert len(products) == 200
        assert len(cat_loader.get_unique_products()) > 0

    if VALIDATION_FILE.exists():
        val_loader = ValidationLoader(VALIDATION_FILE)
        pairs = val_loader.load()
        assert len(pairs) == 330
        assert len(val_loader.get_all_scenes()) == 321
        assert len(val_loader.get_all_products()) == 206
