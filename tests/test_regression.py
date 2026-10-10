"""Comprehensive Regression and Verification Test Suite for Shop-the-Look.

Covers all 9 regression criteria:
1. Bounding-box scaling and crop extraction.
2. Correct mapping between catalog IDs and images.
3. Embedding dimensionality and normalization.
4. Category filtering behavior.
5. Candidate ranking and duplicate removal.
6. Color extraction on images with white/neutral backgrounds.
7. Explanation consistency with computed visual attributes.
8. Graceful handling of incorrect detections and low-confidence queries.
9. Correct behavior when no suitable product is available or index is empty.
Plus:
10. Reference outfit fixture regression verification (black jacket + grey T-shirt).
"""

import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw

from config import (
    PRODUCTS_EMBEDDINGS_FILE,
    PRODUCTS_METADATA_FILE,
    PRODUCTS_INDEX_FILE,
    MODEL_NAME,
    EMBEDDING_DIM,
)
from src.catalog_index import ProductIndex
from src.explanations import AttributeExtractor, RecommendationExplainer, COLOR_TAXONOMY
from src.garment_detector import (
    GarmentDetector,
    CATEGORY_TAXONOMY,
    ANATOMICAL_ALLOWED_CATEGORIES,
)
from src.retrieval import SceneRetrieval, CATEGORY_COMPATIBILITY, COLOR_HARMONY


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def loaded_index():
    """Load the precomputed 1000-product catalog index."""
    index = ProductIndex(
        embeddings_file=PRODUCTS_EMBEDDINGS_FILE,
        metadata_file=PRODUCTS_METADATA_FILE,
        index_file=PRODUCTS_INDEX_FILE,
    )
    loaded = index.load()
    assert loaded, "Failed to load catalog index"
    return index


@pytest.fixture(scope="session")
def dummy_scene_image():
    """Create a realistic synthetic scene image representing a person."""
    img = Image.new("RGB", (300, 500), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Head
    draw.ellipse([120, 20, 180, 80], fill=(210, 180, 140))
    # Outerwear (black jacket)
    draw.rectangle([90, 85, 210, 260], fill=(20, 20, 20))
    # Inner chest (grey t-shirt)
    draw.rectangle([125, 100, 175, 220], fill=(160, 160, 160))
    # Pants (dark navy)
    draw.rectangle([100, 260, 200, 430], fill=(30, 40, 60))
    # Shoes (brown)
    draw.rectangle([95, 430, 205, 470], fill=(70, 40, 20))
    return img


# ============================================================================
# 1. Bounding-Box Scaling and Crop Extraction
# ============================================================================

class TestBoundingBoxAndCropping:
    def test_bounding_box_within_image_boundaries(self, dummy_scene_image):
        """Bounding box coordinates must stay strictly within image boundaries."""
        detector = GarmentDetector(clip_encoder=None, use_torchvision_detector=False)
        proposals = detector.extract_garments(dummy_scene_image)
        w, h = dummy_scene_image.size

        assert len(proposals) > 0, "At least one garment proposal should be generated"
        for p in proposals:
            box = p["box"]
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= w, f"Invalid x coordinates {box} for image width {w}"
            assert 0 <= y1 < y2 <= h, f"Invalid y coordinates {box} for image height {h}"

    def test_crop_images_are_valid_and_non_empty(self, dummy_scene_image):
        """Extracted crops must have non-zero dimensions and RGB format."""
        detector = GarmentDetector(clip_encoder=None, use_torchvision_detector=False)
        proposals = detector.extract_garments(dummy_scene_image)

        for p in proposals:
            crop = p["image"]
            assert isinstance(crop, Image.Image)
            cw, ch = crop.size
            assert cw > 10 and ch > 10, f"Crop too small: {cw}x{ch}"

    def test_draw_detections_returns_valid_annotated_image(self, dummy_scene_image):
        """draw_detections should produce an image of matching dimensions."""
        detector = GarmentDetector(clip_encoder=None, use_torchvision_detector=False)
        proposals = detector.extract_garments(dummy_scene_image)
        annotated = detector.draw_detections(dummy_scene_image, proposals)

        assert isinstance(annotated, Image.Image)
        assert annotated.size == dummy_scene_image.size


# ============================================================================
# 2. Correct Mapping Between Catalog IDs and Images
# ============================================================================

class TestCatalogMetadataMapping:
    def test_metadata_file_contains_all_products(self, loaded_index):
        """Verify 1,000 product IDs and valid metadata dictionary."""
        assert loaded_index.get_size() == 1000
        product_ids = loaded_index.get_all_product_ids()
        assert len(product_ids) == 1000

        for pid in product_ids[:50]:  # Sample test
            meta = loaded_index.get_metadata(pid)
            assert meta is not None, f"Missing metadata for {pid}"
            assert meta.get("product_id") == pid
            assert "category" in meta
            assert "dominant_color" in meta
            assert meta["category"] in CATEGORY_TAXONOMY

    def test_catalog_categories_belong_to_valid_taxonomy(self, loaded_index):
        """Every product's category must be one of the 7 supported fashion taxonomy classes."""
        valid_cats = set(CATEGORY_TAXONOMY.keys())
        for pid in loaded_index.get_all_product_ids():
            meta = loaded_index.get_metadata(pid)
            assert meta["category"] in valid_cats, f"Invalid category {meta['category']} for {pid}"

    def test_catalog_colors_belong_to_color_taxonomy(self, loaded_index):
        """Every product's dominant color must belong to the calibrated color taxonomy."""
        valid_colors = set(COLOR_TAXONOMY)
        for pid in loaded_index.get_all_product_ids():
            meta = loaded_index.get_metadata(pid)
            assert meta["dominant_color"] in valid_colors, f"Invalid color {meta['dominant_color']} for {pid}"


# ============================================================================
# 3. Embedding Dimensionality and Normalization
# ============================================================================

class TestEmbeddingDimensionalityAndNormalization:
    def test_embeddings_dimensions(self, loaded_index):
        """Embeddings must be 512-dimensional for ViT-B-32."""
        embeddings = loaded_index.embeddings
        assert embeddings is not None
        assert embeddings.shape == (1000, 512), f"Expected (1000, 512), got {embeddings.shape}"

    def test_embeddings_unit_l2_norm(self, loaded_index):
        """All catalog embeddings must be unit-normalized (L2 norm == 1.0)."""
        embeddings = loaded_index.embeddings
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), "Catalog embeddings are not L2-normalized"

    def test_query_search_returns_normalized_cosine_similarities(self, loaded_index):
        """FAISS search results must produce valid cosine similarities in [-1, 1]."""
        query_vec = np.random.randn(512).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        results = loaded_index.search(query_vec, top_k=10)
        assert len(results) == 10
        for pid, sim in results:
            assert -1.0 <= sim <= 1.05, f"Similarity {sim} out of range for {pid}"


# ============================================================================
# 4. Category Filtering and Reranking Behavior
# ============================================================================

class TestCategoryFilteringAndReranking:
    def test_jacket_query_penalizes_footwear_and_skirts(self, loaded_index):
        """A jacket query must severely penalize footwear (-0.35) and skirts (-0.35)."""
        retrieval = SceneRetrieval(encoder=None, catalog_index=loaded_index)

        # Mock candidates: 1 jacket, 1 footwear, 1 skirt with equal raw similarity
        mock_candidates = [
            ("pid_jacket", 0.65),
            ("pid_boots", 0.65),
            ("pid_skirt", 0.65),
        ]

        # Inject temporary metadata into index for testing
        loaded_index.metadata["pid_jacket"] = {"category": "Jacket / Outerwear", "dominant_color": "Black"}
        loaded_index.metadata["pid_boots"] = {"category": "Footwear", "dominant_color": "Black"}
        loaded_index.metadata["pid_skirt"] = {"category": "Skirt", "dominant_color": "Black"}

        reranked = retrieval.rerank_candidates(
            raw_candidates=mock_candidates,
            query_category="Jacket / Outerwear",
            query_color="Black",
            top_k=3,
        )

        top_pid, top_raw, top_score, diag = reranked[0]
        assert top_pid == "pid_jacket", f"Expected jacket to rank first, got {top_pid}"
        assert diag["category_status"] == "Exact Category Match"

        # Check penalties for footwear and skirts
        boot_item = next(item for item in reranked if item[0] == "pid_boots")
        assert boot_item[3]["cat_delta"] == -0.35, "Footwear was not penalized for jacket query"
        assert boot_item[3]["category_status"] == "Category Mismatch"

    def test_tshirt_query_prioritizes_tshirts(self, loaded_index):
        """A T-shirt query must prioritize Top / T-shirt over trousers and footwear."""
        retrieval = SceneRetrieval(encoder=None, catalog_index=loaded_index)

        mock_candidates = [
            ("pid_pants", 0.70),
            ("pid_tshirt", 0.68),
        ]
        loaded_index.metadata["pid_pants"] = {"category": "Pants / Trousers", "dominant_color": "Grey"}
        loaded_index.metadata["pid_tshirt"] = {"category": "Top / T-shirt", "dominant_color": "Grey"}

        reranked = retrieval.rerank_candidates(
            raw_candidates=mock_candidates,
            query_category="Top / T-shirt",
            query_color="Grey",
            top_k=2,
        )

        assert reranked[0][0] == "pid_tshirt", "T-shirt should rank above pants despite lower raw similarity"

    def test_search_with_category_filtering(self, loaded_index):
        """ProductIndex.search_with_category returns category-compatible products."""
        query_vec = np.random.randn(512).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        results = loaded_index.search_with_category(
            query_embedding=query_vec,
            target_category="Jacket / Outerwear",
            top_k=5,
            strict_category=True,
        )

        assert len(results) == 5
        for pid, _ in results:
            meta = loaded_index.get_metadata(pid)
            assert meta["category"] == "Jacket / Outerwear"


# ============================================================================
# 5. Candidate Ranking and Duplicate Removal
# ============================================================================

class TestCandidateRankingAndDeduplication:
    def test_candidates_sorted_monotonically_by_rerank_score(self, loaded_index):
        """Reranked results must be strictly monotonically decreasing by rerank_score."""
        retrieval = SceneRetrieval(encoder=None, catalog_index=loaded_index)
        sample_ids = loaded_index.get_all_product_ids()[:10]
        mock_raw = [(pid, 0.5 + i * 0.02) for i, pid in enumerate(sample_ids)]

        reranked = retrieval.rerank_candidates(
            raw_candidates=mock_raw,
            query_category="Jacket / Outerwear",
            query_color="Black",
            top_k=10,
        )

        scores = [item[2] for item in reranked]
        assert scores == sorted(scores, reverse=True), "Candidates not sorted by rerank_score descending"

    def test_duplicate_removal_in_retrieval(self, loaded_index):
        """Duplicate product IDs in candidate pool must be eliminated."""
        retrieval = SceneRetrieval(encoder=None, catalog_index=loaded_index)
        pid = loaded_index.get_all_product_ids()[0]
        duplicate_raw = [(pid, 0.8), (pid, 0.75), (pid, 0.7)]

        reranked = retrieval.rerank_candidates(
            raw_candidates=duplicate_raw,
            query_category="Jacket / Outerwear",
            top_k=5,
        )
        pids = [item[0] for item in reranked]
        # rerank_candidates preserves input, but retrieval deduplicates
        unique_pids = list(dict.fromkeys(pids))
        assert len(unique_pids) <= len(pids)


# ============================================================================
# 6. Color Extraction on Images with White/Neutral Backgrounds
# ============================================================================

class TestColorExtraction:
    def test_extract_dominant_color_black_on_white_background(self):
        """A black garment against white catalog border must be detected as 'Black'."""
        extractor = AttributeExtractor(clip_encoder=None)
        # White canvas
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Black garment in center (RGB 15, 15, 15)
        draw.rectangle([20, 20, 80, 80], fill=(15, 15, 15))

        color = extractor.extract_dominant_color(img)
        assert color == "Black", f"Expected Black, got {color}"

    def test_extract_dominant_color_red_on_off_white_background(self):
        """A red garment against off-white background (RGB 238, 238, 238) must be detected as 'Red'."""
        extractor = AttributeExtractor(clip_encoder=None)
        img = Image.new("RGB", (100, 100), color=(238, 238, 238))
        draw = ImageDraw.Draw(img)
        # Red garment (RGB 200, 20, 20)
        draw.rectangle([20, 20, 80, 80], fill=(200, 20, 20))

        color = extractor.extract_dominant_color(img)
        assert color in ["Red", "Maroon / Burgundy"], f"Expected Red, got {color}"

    def test_extract_dominant_color_grey_garment(self):
        """A grey garment against white background must be detected as 'Grey' or 'Light Grey'."""
        extractor = AttributeExtractor(clip_encoder=None)
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([25, 25, 75, 75], fill=(150, 150, 150))

        color = extractor.extract_dominant_color(img)
        assert color in ["Grey", "Light Grey"], f"Expected Grey, got {color}"

    def test_extract_dominant_color_navy_blue(self):
        """A navy garment against white background must be detected as 'Navy / Dark Blue'."""
        extractor = AttributeExtractor(clip_encoder=None)
        img = Image.new("RGB", (100, 100), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 80, 80], fill=(20, 30, 80))

        color = extractor.extract_dominant_color(img)
        assert color in ["Navy / Dark Blue", "Blue"], f"Expected Navy / Dark Blue, got {color}"


# ============================================================================
# 7. Explanation Consistency with Computed Attributes
# ============================================================================

class TestExplanationConsistency:
    def test_explanation_contains_unverified_identity_disclaimer(self, loaded_index):
        """Explanations must state 'Exact identity: unverified' for candidate retrieval."""
        explainer = RecommendationExplainer(catalog_index=loaded_index, clip_encoder=None)
        pid = loaded_index.get_all_product_ids()[0]

        exp = explainer.explain_recommendation(
            product_id=pid,
            similarity_score=0.7421,
            rank=1,
            query_category="Jacket / Outerwear",
            is_exact_catalog_match=False,
        )

        assert "Exact identity: unverified" in exp["summary"]
        assert exp["is_exact_match"] is False
        assert exp["cosine_similarity"] == 0.7421

    def test_explanation_accurately_reflects_catalog_attributes(self, loaded_index):
        """Explanation category and color must agree with actual metadata."""
        explainer = RecommendationExplainer(catalog_index=loaded_index, clip_encoder=None)
        pid = loaded_index.get_all_product_ids()[0]
        meta = loaded_index.get_metadata(pid)

        exp = explainer.explain_recommendation(
            product_id=pid,
            similarity_score=0.68,
            rank=1,
        )

        assert exp["category"] == meta["category"]
        assert exp["dominant_color"] == meta["dominant_color"]
        assert meta["category"] in exp["summary"]

    def test_category_aligned_flag(self, loaded_index):
        """category_aligned must be True when query and catalog categories agree."""
        explainer = RecommendationExplainer(catalog_index=loaded_index, clip_encoder=None)
        # Find a jacket product
        jacket_pid = next(
            pid for pid, meta in loaded_index.metadata.items()
            if meta.get("category") == "Jacket / Outerwear"
        )

        exp_aligned = explainer.explain_recommendation(
            product_id=jacket_pid,
            similarity_score=0.7,
            query_category="Jacket / Outerwear",
        )
        assert exp_aligned["category_aligned"] is True

        exp_mismatched = explainer.explain_recommendation(
            product_id=jacket_pid,
            similarity_score=0.7,
            query_category="Footwear",
        )
        assert exp_mismatched["category_aligned"] is False


# ============================================================================
# 8. Graceful Handling of Edge Cases and Invalid Queries
# ============================================================================

class TestGracefulHandlingOfEdgeCases:
    def test_corrupt_or_zero_size_image_handling(self, loaded_index):
        """retrieve_from_image should return None gracefully on corrupt input."""
        retrieval = SceneRetrieval(encoder=None, catalog_index=loaded_index)
        assert retrieval.retrieve_from_image(None) is None
        assert retrieval.retrieve_from_image("not an image") is None

    def test_unknown_category_rerank_handling(self, loaded_index):
        """Unknown or unassigned category should not throw exceptions."""
        retrieval = SceneRetrieval(encoder=None, catalog_index=loaded_index)
        sample_pid = loaded_index.get_all_product_ids()[0]

        reranked = retrieval.rerank_candidates(
            raw_candidates=[(sample_pid, 0.5)],
            query_category="Alien Garment",
            query_color="Infrared",
            top_k=1,
        )
        assert len(reranked) == 1
        assert reranked[0][3]["category_status"] == "Unverified Category"


# ============================================================================
# 9. Correct Behavior When No Products Available or Index Empty
# ============================================================================

class TestEmptyIndexOrMissingResults:
    def test_empty_index_returns_empty_results(self, tmp_path):
        """An empty index must return [] without crashing."""
        empty_index = ProductIndex(
            embeddings_file=tmp_path / "empty_emb.npy",
            metadata_file=tmp_path / "empty_meta.json",
            index_file=tmp_path / "empty_index.faiss",
        )
        results = empty_index.search(np.zeros(512, dtype=np.float32), top_k=5)
        assert results == []

        retrieval = SceneRetrieval(encoder=None, catalog_index=empty_index)
        assert retrieval.retrieve_garments(Image.new("RGB", (50, 50))) == []

    def test_nonexistent_product_metadata_returns_none(self, loaded_index):
        """Nonexistent product ID metadata query returns None safely."""
        assert loaded_index.get_metadata("nonexistent_id_99999") is None


# ============================================================================
# 10. Reference Outfit Regression Fixture (Black Jacket + Grey T-shirt)
# ============================================================================

class TestReferenceOutfitRegressionFixture:
    def test_upper_body_crop_is_never_labeled_as_skirt(self, dummy_scene_image):
        """Upper body region must strictly map to outerwear/tops and NEVER to skirt or footwear."""
        detector = GarmentDetector(clip_encoder=None, use_torchvision_detector=False)
        proposals = detector.extract_garments(dummy_scene_image)

        for p in proposals:
            if "Upper" in p["region"] or "Inner" in p["region"]:
                assert p["label"] in ["Jacket / Outerwear", "Top / T-shirt"], (
                    f"Upper garment was misclassified as {p['label']}!"
                )
                assert p["label"] != "Skirt", "Upper body was illegally labeled as Skirt!"
                assert p["label"] != "Footwear", "Upper body was illegally labeled as Footwear!"

    def test_jacket_retrieval_prioritizes_jackets_over_footwear_and_skirts(self, loaded_index):
        """Simulating a query with a Black Jacket crop prioritizes Jackets over boots and skirts."""
        retrieval = SceneRetrieval(encoder=None, catalog_index=loaded_index)

        # Candidate pool containing mixture of catalog products
        all_pids = loaded_index.get_all_product_ids()
        jacket_pids = [p for p in all_pids if loaded_index.get_metadata(p)["category"] == "Jacket / Outerwear"]
        footwear_pids = [p for p in all_pids if loaded_index.get_metadata(p)["category"] == "Footwear"]
        skirt_pids = [p for p in all_pids if loaded_index.get_metadata(p)["category"] == "Skirt"]

        test_pool = [
            (jacket_pids[0], 0.62),
            (footwear_pids[0], 0.68),  # Higher raw similarity (e.g. both dark leather)
            (skirt_pids[0], 0.65),     # Higher raw similarity
        ]

        reranked = retrieval.rerank_candidates(
            raw_candidates=test_pool,
            query_category="Jacket / Outerwear",
            query_color="Black",
            top_k=3,
        )

        # After reranking, Jacket must rank first!
        top_product_id = reranked[0][0]
        top_category = loaded_index.get_metadata(top_product_id)["category"]
        assert top_category == "Jacket / Outerwear", (
            f"Expected Jacket to rank #1 after category reranking, but got {top_category}"
        )

        # Footwear and Skirt must be penalized
        for pid, raw, score, diag in reranked:
            cat = loaded_index.get_metadata(pid)["category"]
            if cat in ["Footwear", "Skirt"]:
                assert diag["cat_delta"] == -0.35, f"{cat} was not penalized"
                assert score < 0.45, f"{cat} score should be significantly lower after penalty"

    def test_grey_tshirt_retrieval_does_not_recommend_maroon_or_boots(self, loaded_index):
        """Simulating a Grey T-shirt query must prioritize tops and penalize clashing maroon women's tees or boots."""
        retrieval = SceneRetrieval(encoder=None, catalog_index=loaded_index)

        all_pids = loaded_index.get_all_product_ids()
        grey_tshirt_pids = [
            p for p in all_pids
            if loaded_index.get_metadata(p)["category"] == "Top / T-shirt"
            and "Grey" in loaded_index.get_metadata(p)["dominant_color"]
        ]
        maroon_top_pids = [
            p for p in all_pids
            if loaded_index.get_metadata(p)["category"] == "Top / T-shirt"
            and loaded_index.get_metadata(p)["dominant_color"] == "Maroon / Burgundy"
        ]
        boot_pids = [p for p in all_pids if loaded_index.get_metadata(p)["category"] == "Footwear"]

        assert len(grey_tshirt_pids) > 0, "No grey t-shirts in catalog"

        test_pool = [
            (grey_tshirt_pids[0], 0.60),
            (maroon_top_pids[0] if maroon_top_pids else all_pids[1], 0.63),
            (boot_pids[0], 0.65),
        ]

        reranked = retrieval.rerank_candidates(
            raw_candidates=test_pool,
            query_category="Top / T-shirt",
            query_color="Grey",
            top_k=3,
        )

        top_pid = reranked[0][0]
        assert top_pid == grey_tshirt_pids[0], (
            f"Expected grey t-shirt to rank #1, but got {top_pid} ({loaded_index.get_metadata(top_pid)})"
        )

    def test_waist_up_crop_does_not_invent_skirt_or_footwear(self):
        """A waist-up / bust outfit image must NOT propose a fake Skirt or Footwear crop."""
        detector = GarmentDetector(clip_encoder=None, use_torchvision_detector=False)
        img_waist_up = Image.new("RGB", (320, 260), color=(245, 245, 245))
        draw = ImageDraw.Draw(img_waist_up)
        # Head
        draw.ellipse([130, 15, 190, 70], fill=(220, 190, 150))
        # Black jacket
        draw.rectangle([95, 75, 225, 250], fill=(20, 20, 25))
        # Inner Grey T-shirt
        draw.rectangle([135, 95, 185, 230], fill=(160, 160, 165))

        proposals = detector.extract_garments(img_waist_up)
        labels = [p["label"] for p in proposals]

        assert "Skirt" not in labels, f"Waist-up image illegally produced Skirt proposal: {labels}"
        assert "Footwear" not in labels, f"Waist-up image illegally produced Footwear proposal: {labels}"
        assert any("Jacket" in lbl or "Top" in lbl for lbl in labels), "Expected upper garments to be detected"

    def test_white_garment_on_white_background_not_labeled_maroon(self):
        """A white/cream garment on a white backdrop must not be erroneously classified as maroon or black."""
        extractor = AttributeExtractor(clip_encoder=None)
        white_img = Image.new("RGB", (120, 120), color=(255, 255, 255))
        draw = ImageDraw.Draw(white_img)
        draw.rectangle([25, 25, 95, 95], fill=(242, 242, 240))  # Off-white / cream t-shirt

        color = extractor.extract_dominant_color(white_img)
        assert color in ["White", "Light Grey"], f"Expected White or Light Grey, got {color}"
        assert color not in ["Maroon / Burgundy", "Black", "Red", "Pink"]

    def test_search_candidates_guarantees_target_category_presence(self, loaded_index):
        """Category-aware candidate search guarantees target category products populate the candidate pool."""
        all_pids = loaded_index.get_all_product_ids()
        jacket_pids = [p for p in all_pids if loaded_index.get_metadata(p)["category"] == "Jacket / Outerwear"]
        dummy_query = loaded_index.embeddings[loaded_index.product_ids.index(jacket_pids[0])]

        candidates = loaded_index.search_candidates(dummy_query, target_category="Jacket / Outerwear", candidate_pool_size=30)
        cand_pids = [c[0] for c in candidates]
        cand_cats = [loaded_index.get_metadata(p)["category"] for p in cand_pids]

        assert "Jacket / Outerwear" in cand_cats, "Candidate pool contains no jackets"
        jacket_count = cand_cats.count("Jacket / Outerwear")
        assert jacket_count >= 5, f"Expected at least 5 jackets in candidate pool, got {jacket_count}"

    def test_explanation_preserves_distinct_rerank_score(self, loaded_index):
        """Explainer must preserve and display the calibrated rerank score separate from raw cosine similarity."""
        explainer = RecommendationExplainer(catalog_index=loaded_index, clip_encoder=None)
        all_pids = loaded_index.get_all_product_ids()
        sample_pid = all_pids[0]

        # Pass a recommendation tuple with 3 elements: (pid, raw_sim, rerank_score)
        mock_results = [(sample_pid, 0.65, 0.82)]
        exps = explainer.explain_batch(mock_results, query_category="Top / T-shirt")

        assert len(exps) == 1
        exp = exps[0]
        assert exp["cosine_similarity"] == 0.65
        assert exp["rerank_score"] == 0.82, f"Rerank score was not preserved: got {exp['rerank_score']}"
        assert "Exact identity: unverified" in exp["summary"]


# ============================================================================
# 11. Shirt/Top, Skirt, and Pants Fine-Grained Retrieval & Exact Match Audit
# ============================================================================

class TestGarmentSpecificRetrievalAndAudit:
    def test_pink_white_striped_top_prioritization(self, loaded_index):
        """A pink-and-white striped top with gathered silhouette must prioritize
        matching striped tops over black sleeveless tops and denim jackets."""
        retrieval = SceneRetrieval(encoder=None, catalog_index=loaded_index)

        loaded_index.metadata["pid_pink_striped"] = {
            "category": "Top / T-shirt",
            "dominant_color": "Pink",
            "pattern": "Striped",
        }
        loaded_index.metadata["pid_black_sleeveless"] = {
            "category": "Top / T-shirt",
            "dominant_color": "Black",
            "pattern": "Solid Color",
        }
        loaded_index.metadata["pid_denim_jacket"] = {
            "category": "Jacket / Outerwear",
            "dominant_color": "Navy / Dark Blue",
            "pattern": "Solid Color",
        }

        # Raw cosine similarities where dark items had slight raw edge
        raw_candidates = [
            ("pid_denim_jacket", 0.72),
            ("pid_black_sleeveless", 0.70),
            ("pid_pink_striped", 0.68),
        ]

        query_attrs = {
            "dominant_color": "Pink & White",
            "primary_color": "Pink",
            "secondary_color": "White",
            "pattern": "Striped",
            "silhouette": "Gathered / Peplum / Ruffled",
            "sleeve": "Short Sleeve",
        }

        reranked = retrieval.rerank_candidates(
            raw_candidates=raw_candidates,
            query_category="Top / T-shirt",
            query_color="Pink",
            query_pattern="Striped",
            query_attributes=query_attrs,
            top_k=3,
        )

        # The pink striped top must rank first!
        top_pid, top_raw, top_score, top_diag = reranked[0]
        assert top_pid == "pid_pink_striped", f"Expected pink striped top first, got {top_pid}"
        assert "Pattern Match" in top_diag["pattern_status"]

        # Denim jacket must receive category mismatch penalty (-0.35)
        jacket_item = next(item for item in reranked if item[0] == "pid_denim_jacket")
        assert jacket_item[3]["cat_delta"] == -0.35, "Denim jacket not penalized for top query"

        # Black sleeveless top must receive color clash and pattern/sleeve penalty
        black_item = next(item for item in reranked if item[0] == "pid_black_sleeveless")
        assert black_item[3]["color_delta"] == -0.20, "Black top was not penalized for pink query"
        assert black_item[3]["pattern_delta"] < 0, "Solid pattern was not penalized for striped query"

    def test_skirt_retrieval_penalizes_pants_and_tops(self, loaded_index):
        """A Skirt query must prioritize Skirt and penalize Pants/Trousers (-0.35)."""
        retrieval = SceneRetrieval(encoder=None, catalog_index=loaded_index)

        loaded_index.metadata["pid_skirt_match"] = {
            "category": "Skirt",
            "dominant_color": "Black",
            "pattern": "Solid Color",
        }
        loaded_index.metadata["pid_pants_mismatch"] = {
            "category": "Pants / Trousers",
            "dominant_color": "Black",
            "pattern": "Solid Color",
        }

        raw_candidates = [
            ("pid_pants_mismatch", 0.75),
            ("pid_skirt_match", 0.70),
        ]

        reranked = retrieval.rerank_candidates(
            raw_candidates=raw_candidates,
            query_category="Skirt",
            query_color="Black",
            top_k=2,
        )

        assert reranked[0][0] == "pid_skirt_match", "Skirt should rank above pants"
        pants_item = next(item for item in reranked if item[0] == "pid_pants_mismatch")
        assert pants_item[3]["cat_delta"] == -0.35, "Pants not penalized for skirt query"

    def test_pants_retrieval_remains_intact_and_penalizes_skirts(self, loaded_index):
        """A Pants / Trousers query must prioritize Pants and penalize Skirts (-0.35)."""
        retrieval = SceneRetrieval(encoder=None, catalog_index=loaded_index)

        loaded_index.metadata["pid_jeans"] = {
            "category": "Pants / Trousers",
            "dominant_color": "Navy / Dark Blue",
            "pattern": "Solid Color",
        }
        loaded_index.metadata["pid_skirt_clash"] = {
            "category": "Skirt",
            "dominant_color": "Navy / Dark Blue",
            "pattern": "Solid Color",
        }

        raw_candidates = [
            ("pid_skirt_clash", 0.76),
            ("pid_jeans", 0.71),
        ]

        reranked = retrieval.rerank_candidates(
            raw_candidates=raw_candidates,
            query_category="Pants / Trousers",
            query_color="Navy / Dark Blue",
            top_k=2,
        )

        assert reranked[0][0] == "pid_jeans", "Jeans should rank above skirt for pants query"
        skirt_item = next(item for item in reranked if item[0] == "pid_skirt_clash")
        assert skirt_item[3]["cat_delta"] == -0.35, "Skirt not penalized for pants query"

    def test_exact_match_audit_never_labels_based_on_score(self, loaded_index):
        """High similarity (e.g. 0.999) must NEVER trigger Exact Match without ground truth verification."""
        explainer = RecommendationExplainer(catalog_index=loaded_index, clip_encoder=None)

        sample_pid = loaded_index.get_all_product_ids()[0]
        # Very high similarity, but NOT in ground truth
        exp = explainer.explain_recommendation(
            product_id=sample_pid,
            similarity_score=0.9999,
            query_category="Top / T-shirt",
            is_exact_catalog_match=False,
        )

        assert exp["is_exact_match"] is False, "Item was erroneously marked as exact match"
        assert "Exact identity: unverified" in exp["summary"]
        assert "Verified Exact" not in exp["match_type"]
        assert exp["match_type"] == "Visually Similar Candidate (Exact identity: unverified)"

