import logging
import json
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
import numpy as np

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluates information retrieval performance with explicit Recall@K and MRR definitions."""

    def __init__(self, recalls_at: Optional[List[int]] = None):
        """Initialize evaluator.

        Args:
            recalls_at: Cutoffs K for Recall@K (default: [1, 5, 10])
        """
        self.recalls_at = recalls_at or [1, 5, 10]
        self.results = []
        self.metrics = {}

    def evaluate_scene(
        self,
        scene_id: str,
        ground_truth_products: Set[str],
        retrieved_products: List[Tuple[str, float]],
    ) -> Optional[dict]:
        """Evaluate retrieval ranking for a single scene against its ground-truth positive set.

        Definition:
        - Relevant Set R: The set of known ground-truth catalog products for this scene.
        - Denominator for Recall@K: |R| (the total number of eligible relevant products for this scene).
        - Recall@K: |TopK ∩ R| / |R|.
        - Reciprocal Rank (RR): 1 / rank of first retrieved relevant item in 1-based ranking, or 0 if none retrieved.

        Args:
            scene_id: Scene ID string
            ground_truth_products: Set of relevant product IDs
            retrieved_products: Ranked list of (product_id, similarity) tuples

        Returns:
            Dict of per-scene metrics
        """
        if not ground_truth_products:
            return None

        retrieved_ids = [pid for pid, _ in retrieved_products]
        gt_set = set(ground_truth_products)
        num_gt = len(gt_set)

        # First relevant product rank (1-indexed) for Reciprocal Rank
        reciprocal_rank = 0.0
        first_hit_rank = None
        for rank_idx, pid in enumerate(retrieved_ids, 1):
            if pid in gt_set:
                reciprocal_rank = 1.0 / rank_idx
                first_hit_rank = rank_idx
                break

        # Recall@K calculation
        recalls = {}
        hits = {}
        for k in self.recalls_at:
            top_k_ids = set(retrieved_ids[:k])
            hit_count = len(top_k_ids & gt_set)
            recall_k = hit_count / num_gt if num_gt > 0 else 0.0
            recalls[f"recall@{k}"] = round(recall_k, 5)
            hits[f"hits@{k}"] = hit_count

        result = {
            "scene_id": scene_id,
            "num_ground_truth": num_gt,
            "num_retrieved": len(retrieved_ids),
            "first_hit_rank": first_hit_rank,
            "reciprocal_rank": round(reciprocal_rank, 5),
            "retrieved_top_5": retrieved_ids[:5],
            "ground_truth_ids": list(gt_set),
            **recalls,
            **hits,
        }

        self.results.append(result)
        return result

    def compute_metrics(self) -> Dict:
        """Compute macro-averaged aggregate metrics across all evaluated scenes.

        Returns:
            Dict containing mean, std, min, and max for MRR and each Recall@K.
        """
        if not self.results:
            return {}

        rr_values = [r["reciprocal_rank"] for r in self.results]
        metrics = {
            "num_evaluated_scenes": len(self.results),
            "mrr": float(np.mean(rr_values)),
            "mrr_std": float(np.std(rr_values)),
        }

        for k in self.recalls_at:
            key = f"recall@{k}"
            vals = [r[key] for r in self.results]
            metrics[key] = float(np.mean(vals))
            metrics[f"{key}_std"] = float(np.std(vals))
            metrics[f"{key}_min"] = float(np.min(vals))
            metrics[f"{key}_max"] = float(np.max(vals))

        self.metrics = metrics
        return metrics

    def get_results(self) -> List[Dict]:
        """Return raw list of per-scene evaluation result dictionaries."""
        return self.results.copy()

    def save_results(self, json_path: Path, csv_path: Optional[Path] = None):
        """Save results to JSON and optionally CSV."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "aggregate_metrics": self.metrics,
            "per_scene_evaluations": self.results,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved evaluation JSON to {json_path}")

        if csv_path:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            if self.results:
                fields = [
                    "scene_id",
                    "num_ground_truth",
                    "first_hit_rank",
                    "reciprocal_rank",
                ] + [f"recall@{k}" for k in self.recalls_at] + ["ground_truth_ids", "retrieved_top_5"]

                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                    for r in self.results:
                        writer.writerow({
                            **r,
                            "ground_truth_ids": ";".join(r["ground_truth_ids"]),
                            "retrieved_top_5": ";".join(r["retrieved_top_5"]),
                        })
                logger.info(f"Saved evaluation CSV to {csv_path}")

    def print_metrics(self):
        """Print formatted metrics to console."""
        if not self.metrics:
            print("No metrics computed.")
            return

        print("\n" + "=" * 60)
        print("RETRIEVAL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Evaluated Scenes:       {self.metrics['num_evaluated_scenes']}")
        print(f"Mean Reciprocal Rank:   {self.metrics['mrr']:.4f} (std: {self.metrics['mrr_std']:.4f})")
        for k in self.recalls_at:
            key = f"recall@{k}"
            mean = self.metrics.get(key, 0.0)
            std = self.metrics.get(f"{key}_std", 0.0)
            print(f"Recall@{k:<2}:                {mean:.4f} (std: {std:.4f})")
        print("=" * 60)


class EvaluationPipeline:
    """End-to-end evaluation pipeline with rigorous coverage reporting."""

    def __init__(
        self,
        image_downloader,
        image_encoder,
        catalog_index,
        recalls_at: Optional[List[int]] = None,
    ):
        self.image_downloader = image_downloader
        self.image_encoder = image_encoder
        self.catalog_index = catalog_index
        self.recalls_at = recalls_at or [1, 5, 10]
        self.evaluator = RetrievalEvaluator(self.recalls_at)

        self.coverage_stats = {
            "total_validation_pairs": 0,
            "unique_scenes": 0,
            "scene_images_accessible": 0,
            "scene_images_unavailable": 0,
            "unique_ground_truth_products": 0,
            "gt_products_in_catalog": 0,
            "gt_products_missing_from_catalog": 0,
            "gt_products_with_embeddings": 0,
            "eligible_scenes_evaluated": 0,
            "scenes_skipped_no_image": 0,
            "scenes_skipped_no_indexed_gt": 0,
        }

    def evaluate(self, validation_pairs: List[Dict], scene_to_products: Dict[str, List[str]]) -> Dict:
        """Run evaluation across validation scenes with thorough data integrity auditing.

        Args:
            validation_pairs: List of {"scene": ..., "product": ...} dicts
            scene_to_products: Dict mapping scene_id -> list of product IDs

        Returns:
            Dict containing evaluation metrics and coverage statistics
        """
        self.coverage_stats["total_validation_pairs"] = len(validation_pairs)
        self.coverage_stats["unique_scenes"] = len(scene_to_products)

        all_gt_prods = set()
        for prods in scene_to_products.values():
            all_gt_prods.update(prods)
        self.coverage_stats["unique_ground_truth_products"] = len(all_gt_prods)

        indexed_pids = set(self.catalog_index.get_all_product_ids())

        # Check catalog and index presence
        gt_in_index = all_gt_prods & indexed_pids
        self.coverage_stats["gt_products_with_embeddings"] = len(gt_in_index)

        max_k = max(self.recalls_at)

        for scene_id, gt_prods in scene_to_products.items():
            # 1. Check scene image
            scene_img_path = self.image_downloader.get_cached_image(scene_id)
            if scene_img_path is None or not scene_img_path.exists():
                self.coverage_stats["scene_images_unavailable"] += 1
                self.coverage_stats["scenes_skipped_no_image"] += 1
                continue
            self.coverage_stats["scene_images_accessible"] += 1

            # 2. Check which ground truth products are available in the indexed catalog
            eligible_gt = set(p for p in gt_prods if p in indexed_pids)
            if not eligible_gt:
                self.coverage_stats["scenes_skipped_no_indexed_gt"] += 1
                continue

            # 3. Encode scene image
            scene_emb = self.image_encoder.encode_image(scene_img_path)
            if scene_emb is None:
                continue

            # 4. Retrieve top-K candidates from catalog index
            retrieved = self.catalog_index.search(scene_emb, top_k=max_k)

            # Deduplicate retrieved candidates
            seen = set()
            deduped = []
            for pid, sim in retrieved:
                if pid not in seen:
                    seen.add(pid)
                    deduped.append((pid, sim))

            # 5. Evaluate scene against its eligible ground-truth items
            self.evaluator.evaluate_scene(
                scene_id=scene_id,
                ground_truth_products=eligible_gt,
                retrieved_products=deduped,
            )

        self.coverage_stats["eligible_scenes_evaluated"] = len(self.evaluator.results)
        metrics = self.evaluator.compute_metrics()

        return {
            "metrics": metrics,
            "coverage": self.coverage_stats,
        }

    def print_summary(self):
        """Print full evaluation summary including coverage statistics."""
        print("\n" + "=" * 60)
        print("EVALUATION & COVERAGE AUDIT SUMMARY")
        print("=" * 60)
        print(f"Total Validation Pairs:          {self.coverage_stats['total_validation_pairs']}")
        print(f"Unique Validation Scenes:        {self.coverage_stats['unique_scenes']}")
        print(f"Accessible Scene Images:         {self.coverage_stats['scene_images_accessible']}")
        print(f"Missing Scene Images:            {self.coverage_stats['scene_images_unavailable']}")
        print(f"Unique GT Products:              {self.coverage_stats['unique_ground_truth_products']}")
        print(f"GT Products Indexed in Catalog:  {self.coverage_stats['gt_products_with_embeddings']}")
        print(f"Scenes Skipped (No Image):       {self.coverage_stats['scenes_skipped_no_image']}")
        print(f"Scenes Skipped (No Indexed GT):  {self.coverage_stats['scenes_skipped_no_indexed_gt']}")
        print(f"Eligible Scenes Evaluated:       {self.coverage_stats['eligible_scenes_evaluated']}")
        if self.coverage_stats['unique_scenes'] > 0:
            cov = self.coverage_stats['eligible_scenes_evaluated'] / self.coverage_stats['unique_scenes']
            print(f"Evaluation Scene Coverage:       {cov:.1%}")
        self.evaluator.print_metrics()
