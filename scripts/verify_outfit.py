import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw
import numpy as np
from config import PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_METADATA_FILE, PRODUCTS_INDEX_FILE
from src.catalog_index import ProductIndex
from src.garment_detector import GarmentDetector
from src.retrieval import SceneRetrieval
from src.explanations import RecommendationExplainer

# 1. Load Catalog Index
index = ProductIndex(PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_METADATA_FILE, PRODUCTS_INDEX_FILE)
index.load()
detector = GarmentDetector(clip_encoder=None, use_torchvision_detector=False)
retrieval = SceneRetrieval(encoder=None, catalog_index=index, garment_detector=detector)
explainer = RecommendationExplainer(catalog_index=index, clip_encoder=None)
all_pids = index.get_all_product_ids()

print("=" * 70)
print("VERIFICATION SUITE: SHOP-THE-LOOK PRODUCT RETRIEVAL & EXACT MATCH AUDIT")
print("=" * 70)

# =========================================================================
# Case 1: Pink-and-White Striped Gathered Top (Testing Non-sleeveless, Non-jacket)
# =========================================================================
print("\n=== CASE 1: PINK-AND-WHITE STRIPED TOP WITH GATHERED SILHOUETTE ===")
# Create mock candidate set including:
# - Pink striped top
# - Black sleeveless top
# - Denim jacket
# - White top
index.metadata["pid_pink_striped"] = {
    "product_id": "pid_pink_striped",
    "category": "Top / T-shirt",
    "dominant_color": "Pink",
    "pattern": "Striped",
}
index.metadata["pid_black_sleeveless"] = {
    "product_id": "pid_black_sleeveless",
    "category": "Top / T-shirt",
    "dominant_color": "Black",
    "pattern": "Solid Color",
}
index.metadata["pid_denim_jacket"] = {
    "product_id": "pid_denim_jacket",
    "category": "Jacket / Outerwear",
    "dominant_color": "Navy / Dark Blue",
    "pattern": "Solid Color",
}

mock_top_cands = [
    ("pid_denim_jacket", 0.74),       # Initially higher raw similarity
    ("pid_black_sleeveless", 0.72),   # High raw similarity
    ("pid_pink_striped", 0.69),       # Ground truth / intended item
]

query_attrs_top = {
    "dominant_color": "Pink & White",
    "primary_color": "Pink",
    "secondary_color": "White",
    "pattern": "Striped",
    "silhouette": "Gathered / Peplum / Ruffled",
    "sleeve": "Short Sleeve",
}

reranked_top = retrieval.rerank_candidates(
    mock_top_cands,
    query_category="Top / T-shirt",
    query_color="Pink",
    query_pattern="Striped",
    query_attributes=query_attrs_top,
    top_k=3,
)

print(f"Query Attributes: Color='Pink & White', Pattern='Striped', Silhouette='Gathered', Sleeve='Short Sleeve'")
for rank, (pid, raw, score, diag) in enumerate(reranked_top, 1):
    meta = index.get_metadata(pid)
    exp = explainer.explain_recommendation(
        pid, raw, rerank_score=score, query_category="Top / T-shirt"
    )
    print(f"Rank #{rank}: {pid:<22} | Cat: {meta['category']:<18} | Color: {meta['dominant_color']:<10} | Pat: {meta.get('pattern', ''):<12} | Raw: {raw:.4f} | Rerank: {score:.4f}")
    print(f"  Diagnostics: CatDelta={diag['cat_delta']:+.2f} ({diag['category_status']}), ColorDelta={diag['color_delta']:+.2f} ({diag['color_status']}), PatDelta={diag['pattern_delta']:+.2f} ({diag['pattern_status']}), StyleDelta={diag['style_delta']:+.2f}")
    print(f"  Summary: {exp['summary']}")

assert reranked_top[0][0] == "pid_pink_striped", "Pink striped top must rank first!"
print(">>> PASS: Pink-and-white striped top prioritized; denim jacket & black sleeveless top successfully penalized and demoted.")


# =========================================================================
# Case 2: Skirt Retrieval (Category-Aware Matching vs Pants)
# =========================================================================
print("\n=== CASE 2: SKIRT RETRIEVAL (CATEGORY-AWARE RERANKING) ===")
index.metadata["pid_skirt_match"] = {
    "product_id": "pid_skirt_match",
    "category": "Skirt",
    "dominant_color": "Black",
    "pattern": "Solid Color",
}
index.metadata["pid_pants_bleed"] = {
    "product_id": "pid_pants_bleed",
    "category": "Pants / Trousers",
    "dominant_color": "Black",
    "pattern": "Solid Color",
}

mock_skirt_cands = [
    ("pid_pants_bleed", 0.77),  # Trousers initially higher raw similarity
    ("pid_skirt_match", 0.71),  # Intended skirt
]

reranked_skirt = retrieval.rerank_candidates(
    mock_skirt_cands,
    query_category="Skirt",
    query_color="Black",
    top_k=2,
)

for rank, (pid, raw, score, diag) in enumerate(reranked_skirt, 1):
    meta = index.get_metadata(pid)
    exp = explainer.explain_recommendation(
        pid, raw, rerank_score=score, query_category="Skirt"
    )
    print(f"Rank #{rank}: {pid:<22} | Cat: {meta['category']:<18} | Color: {meta['dominant_color']:<10} | Raw: {raw:.4f} | Rerank: {score:.4f}")
    print(f"  Diagnostics: CatDelta={diag['cat_delta']:+.2f} ({diag['category_status']})")
    print(f"  Summary: {exp['summary']}")

assert reranked_skirt[0][0] == "pid_skirt_match", "Skirt must rank above trousers!"
print(">>> PASS: Skirt retrieval cleanly prioritizes skirts and penalizes trousers.")


# =========================================================================
# Case 3: Pants / Trousers Retrieval (Intact and Unbroken)
# =========================================================================
print("\n=== CASE 3: PANTS / TROUSERS RETRIEVAL (PRESERVING WORKING BEHAVIOR) ===")
index.metadata["pid_jeans_match"] = {
    "product_id": "pid_jeans_match",
    "category": "Pants / Trousers",
    "dominant_color": "Navy / Dark Blue",
    "pattern": "Solid Color",
}
index.metadata["pid_skirt_bleed"] = {
    "product_id": "pid_skirt_bleed",
    "category": "Skirt",
    "dominant_color": "Navy / Dark Blue",
    "pattern": "Solid Color",
}

mock_pants_cands = [
    ("pid_skirt_bleed", 0.78),
    ("pid_jeans_match", 0.73),
]

reranked_pants = retrieval.rerank_candidates(
    mock_pants_cands,
    query_category="Pants / Trousers",
    query_color="Navy / Dark Blue",
    top_k=2,
)

for rank, (pid, raw, score, diag) in enumerate(reranked_pants, 1):
    meta = index.get_metadata(pid)
    exp = explainer.explain_recommendation(
        pid, raw, rerank_score=score, query_category="Pants / Trousers"
    )
    print(f"Rank #{rank}: {pid:<22} | Cat: {meta['category']:<18} | Color: {meta['dominant_color']:<10} | Raw: {raw:.4f} | Rerank: {score:.4f}")
    print(f"  Diagnostics: CatDelta={diag['cat_delta']:+.2f} ({diag['category_status']})")
    print(f"  Summary: {exp['summary']}")

assert reranked_pants[0][0] == "pid_jeans_match", "Jeans must rank above skirt!"
print(">>> PASS: Pants retrieval preserved; skirts penalized as category mismatch.")


# =========================================================================
# Case 4: Exact Match Logic Audit
# =========================================================================
print("\n=== CASE 4: EXACT MATCH LOGIC AUDIT ===")
sample_pid = all_pids[0]
exp_high_sim = explainer.explain_recommendation(
    product_id=sample_pid,
    similarity_score=0.9999,
    query_category="Top / T-shirt",
    is_exact_catalog_match=False,
)
print(f"Product ID: {sample_pid}")
print(f"Raw Cosine Similarity: {exp_high_sim['cosine_similarity']} (extremely high)")
print(f"is_exact_match flag: {exp_high_sim['is_exact_match']}")
print(f"Match Type Badge: '{exp_high_sim['match_type']}'")
print(f"Summary Statement: \"{exp_high_sim['summary']}\"")

assert exp_high_sim["is_exact_match"] is False
assert "Exact identity: unverified" in exp_high_sim["summary"]
print(">>> PASS: Exact match is never granted based on similarity threshold; requires verified benchmark ID.")

print("\n" + "=" * 70)
print("ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY.")
print("=" * 70)
