"""Deep analysis: Which validation products ARE in the index and do they have working images?"""
import json
import numpy as np
from pathlib import Path

# Load index
with open("artifacts/embeddings/products_metadata.json") as f:
    meta = json.load(f)
indexed_pids = set(meta["product_ids"])

# Load validation
val_pairs = []
with open("data/validation.jsonl") as f:
    for line in f:
        d = json.loads(line)
        val_pairs.append(d)

val_products = set(p["product"] for p in val_pairs)
val_scenes = set(p["scene"] for p in val_pairs)

# Check which validation products are indexed
in_index = val_products & indexed_pids
not_in_index = val_products - indexed_pids
print(f"Validation products IN index: {len(in_index)}")
print(f"Validation products NOT in index: {len(not_in_index)}")

# For products in index, check if images exist
images_dir = Path("artifacts/images")
in_index_with_image = 0
in_index_no_image = 0
for pid in in_index:
    m = meta["metadata"].get(pid, {})
    img_path = m.get("image_path")
    if img_path and Path(img_path).exists():
        in_index_with_image += 1
    else:
        in_index_no_image += 1

print(f"\nOf {len(in_index)} indexed validation products:")
print(f"  With image: {in_index_with_image}")
print(f"  Without image: {in_index_no_image}")

# Check scene images
scene_with_image = 0
for sid in val_scenes:
    found = False
    for img_path in images_dir.rglob(f"{sid}.jpg"):
        found = True
        break
    if found:
        scene_with_image += 1

print(f"\nScene images available: {scene_with_image} / {len(val_scenes)}")

# Check if the current low recall is simply because most GT products are missing
# Scene breakdown
scene_to_products = {}
for p in val_pairs:
    if p["scene"] not in scene_to_products:
        scene_to_products[p["scene"]] = set()
    scene_to_products[p["scene"]].add(p["product"])

scenes_with_all_gt_in_index = 0
scenes_with_some_gt_in_index = 0
scenes_with_no_gt_in_index = 0
for sid, gt_products in scene_to_products.items():
    gt_in_index = gt_products & indexed_pids
    if len(gt_in_index) == len(gt_products):
        scenes_with_all_gt_in_index += 1
    elif len(gt_in_index) > 0:
        scenes_with_some_gt_in_index += 1
    else:
        scenes_with_no_gt_in_index += 1

print(f"\nScene GT coverage:")
print(f"  All GT products in index: {scenes_with_all_gt_in_index} / {len(scene_to_products)}")
print(f"  Some GT products in index: {scenes_with_some_gt_in_index} / {len(scene_to_products)}")
print(f"  No GT products in index: {scenes_with_no_gt_in_index} / {len(scene_to_products)}")

# Check build_index.py to see why only 1000 products were indexed
print("\n--- Checking build_index.py for limit ---")
with open("scripts/build_index.py") as f:
    content = f.read()
    if "1000" in content or "MAX_IMAGES" in content or "limit" in content.lower():
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "1000" in line or "limit" in line.lower() or "max" in line.lower():
                print(f"  Line {i+1}: {line.strip()}")
