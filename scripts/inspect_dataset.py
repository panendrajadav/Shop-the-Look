#!/usr/bin/env python3
"""Inspect datasets and verify schemas, counts, relationships, and image URL accessibility."""

import sys
import json
import logging
import requests
from io import BytesIO
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PRODUCT_CATALOG_FILE, VALIDATION_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_to_url(signature: str, scheme: str = "https") -> str:
    """Assignment URL pattern with HTTPS preference and fallback support."""
    return f"{scheme}://i.pinimg.com/400x/{signature[0:2]}/{signature[2:4]}/{signature[4:6]}/{signature}.jpg"


def inspect_catalog(catalog_path: Path):
    logger.info("=" * 60)
    logger.info(f"INSPECTING PRODUCT CATALOG: {catalog_path}")
    logger.info("=" * 60)

    total_records = 0
    valid_records = 0
    corrupt_lines = 0
    missing_id_records = 0
    product_ids = []
    schemas = set()

    with open(catalog_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            total_records += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                schemas.add(tuple(sorted(record.keys())))
                pid = record.get("product")
                if pid:
                    valid_records += 1
                    product_ids.append(pid)
                else:
                    missing_id_records += 1
            except Exception as e:
                corrupt_lines += 1

    unique_product_ids = set(product_ids)
    duplicates = len(product_ids) - len(unique_product_ids)
    counts = Counter(product_ids)
    most_common = counts.most_common(5)

    id_lengths = set(len(pid) for pid in unique_product_ids)

    print(f"Total Lines/Records:        {total_records}")
    print(f"Valid Product Records:      {valid_records}")
    print(f"Unique Product IDs:         {len(unique_product_ids)}")
    print(f"Duplicate Product Entries:  {duplicates}")
    print(f"Corrupted Lines:            {corrupt_lines}")
    print(f"Missing Product Key:        {missing_id_records}")
    print(f"Observed JSON Keys:         {schemas}")
    print(f"Product ID Lengths:         {id_lengths}")
    print(f"Top 5 most frequent IDs:    {most_common}")

    return unique_product_ids


def inspect_validation(val_path: Path, catalog_ids: set):
    logger.info("=" * 60)
    logger.info(f"INSPECTING VALIDATION DATASET: {val_path}")
    logger.info("=" * 60)

    total_pairs = 0
    valid_pairs = 0
    corrupted_lines = 0
    scene_ids = []
    val_product_ids = []
    scene_to_products = defaultdict(list)
    product_to_scenes = defaultdict(list)
    schemas = set()

    with open(val_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            total_pairs += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                schemas.add(tuple(sorted(record.keys())))
                scene = record.get("scene")
                product = record.get("product")
                if scene and product:
                    valid_pairs += 1
                    scene_ids.append(scene)
                    val_product_ids.append(product)
                    scene_to_products[scene].append(product)
                    product_to_scenes[product].append(scene)
                else:
                    corrupted_lines += 1
            except Exception:
                corrupted_lines += 1

    unique_scenes = set(scene_ids)
    unique_val_products = set(val_product_ids)

    prods_in_catalog = unique_val_products & catalog_ids
    prods_missing_from_catalog = unique_val_products - catalog_ids

    # Multiple products per scene analysis
    prods_per_scene = [len(set(prods)) for prods in scene_to_products.values()]
    multi_product_scenes = sum(1 for count in prods_per_scene if count > 1)

    print(f"Total Validation Pairs:     {total_pairs}")
    print(f"Valid Pairs:                {valid_pairs}")
    print(f"Unique Scenes:              {len(unique_scenes)}")
    print(f"Unique Validation Products: {len(unique_val_products)}")
    print(f"Observed Validation Keys:   {schemas}")
    print(f"Scenes with Multiple Products: {multi_product_scenes}")
    print(f"Max Products in Single Scene:  {max(prods_per_scene) if prods_per_scene else 0}")
    print(f"Validation Products in Catalog: {len(prods_in_catalog)} ({len(prods_in_catalog)/len(unique_val_products):.1%})")
    print(f"Validation Products MISSING:    {len(prods_missing_from_catalog)} ({len(prods_missing_from_catalog)/len(unique_val_products):.1%})")

    return unique_scenes, unique_val_products


def test_url_access(sample_ids: list, id_type: str = "product"):
    logger.info("=" * 60)
    logger.info(f"TESTING URL ACCESSIBILITY FOR {id_type.upper()}S (Sample size: {len(sample_ids)})")
    logger.info("=" * 60)

    success_https = 0
    success_http = 0
    failures = 0
    decode_success = 0
    decode_failures = 0

    for item_id in sample_ids:
        # Try HTTPS first
        url_https = convert_to_url(item_id, scheme="https")
        status_code = None
        content = None
        used_scheme = "https"

        try:
            resp = requests.get(url_https, timeout=5)
            status_code = resp.status_code
            if status_code == 200:
                success_https += 1
                content = resp.content
        except Exception:
            pass

        # Fallback to HTTP if HTTPS failed or returned non-200
        if status_code != 200:
            url_http = convert_to_url(item_id, scheme="http")
            used_scheme = "http"
            try:
                resp = requests.get(url_http, timeout=5)
                status_code = resp.status_code
                if status_code == 200:
                    success_http += 1
                    content = resp.content
                else:
                    failures += 1
            except Exception:
                failures += 1

        # Check image decoding if content retrieved
        if content:
            try:
                img = Image.open(BytesIO(content))
                img.verify()
                decode_success += 1
                print(f"[{used_scheme.upper()} 200 OK] {item_id} -> Size: {len(content)} bytes, Valid image")
            except Exception as e:
                decode_failures += 1
                print(f"[{used_scheme.upper()} 200 CORRUPT] {item_id} -> Image decode failed: {e}")
        else:
            print(f"[FAILED {status_code}] {item_id}")

    total = len(sample_ids)
    print("\nSummary:")
    print(f"  Total tested:       {total}")
    print(f"  HTTPS Successful:   {success_https}")
    print(f"  HTTP Fallback Succ: {success_http}")
    print(f"  Total Downloaded:   {success_https + success_http}/{total} ({(success_https + success_http)/total:.1%})")
    print(f"  Valid Decoded:      {decode_success}/{total}")
    print(f"  Decode Failures:    {decode_failures}")


def main():
    catalog_ids = inspect_catalog(PRODUCT_CATALOG_FILE)
    val_scenes, val_products = inspect_validation(VALIDATION_FILE, catalog_ids)

    # Test sample of product IDs
    sample_products = list(catalog_ids)[:5]
    test_url_access(sample_products, id_type="catalog product")

    # Test sample of scene IDs
    sample_scenes = list(val_scenes)[:5]
    test_url_access(sample_scenes, id_type="validation scene")


if __name__ == "__main__":
    main()
