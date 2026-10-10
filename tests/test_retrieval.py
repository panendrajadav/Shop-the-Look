#!/usr/bin/env python3
"""Unit tests for FAISS ProductIndex, embedding normalization, retrieval, and deduplication."""

import sys
import numpy as np
from pathlib import Path
import tempfile
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalog_index import ProductIndex
from src.retrieval import SceneRetrieval


def test_product_index_build_search_and_persistence():
    """Verify FAISS IndexFlatIP build, search accuracy, persistence, and reloading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        embeddings_file = tmpdir / "test_embeddings.npy"
        metadata_file = tmpdir / "test_metadata.json"
        index_file = tmpdir / "test_index.idx"

        # Create 3 normalized vectors
        v1 = np.random.randn(512).astype(np.float32)
        v1 /= np.linalg.norm(v1)

        v2 = np.random.randn(512).astype(np.float32)
        v2 /= np.linalg.norm(v2)

        v3 = np.random.randn(512).astype(np.float32)
        v3 /= np.linalg.norm(v3)

        product_embeddings = {"p1": v1, "p2": v2, "p3": v3}
        product_metadata = {
            "p1": {"product_id": "p1", "color": "black"},
            "p2": {"product_id": "p2", "color": "white"},
            "p3": {"product_id": "p3", "color": "blue"},
        }

        # Build index
        index = ProductIndex(embeddings_file, metadata_file, index_file)
        num_indexed = index.build_index(product_embeddings, product_metadata)
        assert num_indexed == 3
        assert index.get_size() == 3

        # Self-search: query with v1 should return p1 as top-1 with similarity ~ 1.0
        results = index.search(v1, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "p1"
        assert abs(results[0][1] - 1.0) < 1e-4

        # Save and reload
        index.save()
        assert embeddings_file.exists()
        assert metadata_file.exists()
        assert index_file.exists()

        index_reloaded = ProductIndex(embeddings_file, metadata_file, index_file)
        assert index_reloaded.load()
        assert index_reloaded.get_size() == 3

        reloaded_results = index_reloaded.search(v1, top_k=2)
        assert reloaded_results[0][0] == "p1"


def test_empty_catalog_index():
    """Verify empty index returns empty results without crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        index = ProductIndex(tmpdir / "emb.npy", tmpdir / "meta.json", tmpdir / "idx.idx")
        dummy_query = np.zeros(512, dtype=np.float32)
        res = index.search(dummy_query, top_k=5)
        assert res == []


def test_retrieval_deduplication():
    """Verify deduplication removes redundant recommendations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        index = ProductIndex(tmpdir / "emb.npy", tmpdir / "meta.json", tmpdir / "idx.idx")

        # Mock index search returning duplicate IDs
        class MockIndex:
            def get_size(self):
                return 10
            def search(self, emb, top_k=5):
                return [("p1", 0.95), ("p1", 0.94), ("p2", 0.85), ("p3", 0.80)]

        mock_index = MockIndex()
        retrieval = SceneRetrieval(encoder=None, catalog_index=mock_index)

        # With deduplication
        deduped = retrieval.retrieve_from_embedding(np.zeros(512), top_k=3, remove_duplicates=True)
        assert len(deduped) == 3
        pids = [p for p, _ in deduped]
        assert pids == ["p1", "p2", "p3"]
        assert len(set(pids)) == len(pids)

        # Without deduplication
        raw = retrieval.retrieve_from_embedding(np.zeros(512), top_k=3, remove_duplicates=False)
        assert len(raw) == 3
        assert [p for p, _ in raw] == ["p1", "p1", "p2"]
