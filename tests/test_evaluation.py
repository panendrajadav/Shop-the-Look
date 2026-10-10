#!/usr/bin/env python3
"""Unit tests for information retrieval metrics: Recall@K, MRR, denominators, and edge cases."""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation import RetrievalEvaluator


def test_perfect_retrieval():
    """Verify metrics when all ground truth products are retrieved at top ranks."""
    evaluator = RetrievalEvaluator(recalls_at=[1, 5, 10])
    gt_products = {"p1", "p2"}
    retrieved = [("p1", 0.99), ("p2", 0.95), ("p3", 0.80)]

    res = evaluator.evaluate_scene("scene_1", gt_products, retrieved)

    # 1 of 2 retrieved at rank 1 -> Recall@1 = 0.5
    assert res["recall@1"] == 0.5
    # 2 of 2 retrieved at rank 2 -> Recall@5 = 1.0, Recall@10 = 1.0
    assert res["recall@5"] == 1.0
    assert res["recall@10"] == 1.0
    # First relevant at rank 1 -> RR = 1.0
    assert res["reciprocal_rank"] == 1.0


def test_delayed_hit_mrr():
    """Verify MRR calculation when first relevant item appears at rank 3."""
    evaluator = RetrievalEvaluator(recalls_at=[1, 5, 10])
    gt_products = {"p_target"}
    retrieved = [("irrelevant_1", 0.9), ("irrelevant_2", 0.8), ("p_target", 0.7)]

    res = evaluator.evaluate_scene("scene_2", gt_products, retrieved)

    assert res["recall@1"] == 0.0
    assert res["recall@5"] == 1.0
    # First hit is at rank 3 -> RR = 1/3
    assert abs(res["reciprocal_rank"] - 1.0 / 3.0) < 1e-4


def test_zero_hits():
    """Verify metrics when no ground truth products are retrieved."""
    evaluator = RetrievalEvaluator(recalls_at=[1, 5, 10])
    gt_products = {"p_target"}
    retrieved = [("p_other_1", 0.9), ("p_other_2", 0.8), ("p_other_3", 0.7)]

    res = evaluator.evaluate_scene("scene_3", gt_products, retrieved)

    assert res["recall@1"] == 0.0
    assert res["recall@5"] == 0.0
    assert res["recall@10"] == 0.0
    assert res["reciprocal_rank"] == 0.0


def test_macro_averaged_aggregation():
    """Verify macro-average across multiple scenes."""
    evaluator = RetrievalEvaluator(recalls_at=[1, 5])

    # Scene 1: GT={p1}, retrieved at rank 1 -> R@1=1.0, R@5=1.0, RR=1.0
    evaluator.evaluate_scene("s1", {"p1"}, [("p1", 0.9)])

    # Scene 2: GT={p2}, retrieved at rank 2 -> R@1=0.0, R@5=1.0, RR=0.5
    evaluator.evaluate_scene("s2", {"p2"}, [("p_other", 0.9), ("p2", 0.8)])

    metrics = evaluator.compute_metrics()

    assert metrics["num_evaluated_scenes"] == 2
    # Mean Recall@1 = (1.0 + 0.0) / 2 = 0.5
    assert metrics["recall@1"] == 0.5
    # Mean Recall@5 = (1.0 + 1.0) / 2 = 1.0
    assert metrics["recall@5"] == 1.0
    # Mean MRR = (1.0 + 0.5) / 2 = 0.75
    assert metrics["mrr"] == 0.75


def test_empty_ground_truth_ignored():
    """Verify scenes with empty ground truth are safely bypassed."""
    evaluator = RetrievalEvaluator(recalls_at=[1, 5])
    res = evaluator.evaluate_scene("s_empty", set(), [("p1", 0.9)])
    assert res is None
    assert len(evaluator.results) == 0


def test_multiple_ground_truth_denominator():
    """Verify that multiple ground-truth items properly divide the denominator."""
    evaluator = RetrievalEvaluator(recalls_at=[1, 2, 5])
    gt_products = {"p1", "p2", "p3", "p4"}
    retrieved = [("p1", 0.9), ("p2", 0.8)]  # Only 2 out of 4 retrieved

    res = evaluator.evaluate_scene("s_multi", gt_products, retrieved)
    assert res["num_ground_truth"] == 4
    assert res["recall@1"] == 0.25  # 1/4
    assert res["recall@2"] == 0.50  # 2/4
    assert res["recall@5"] == 0.50  # 2/4
