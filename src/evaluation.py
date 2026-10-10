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
    """End-to-end evaluation with coverage reporting and a per-scene breakdown.

    Predictions come from SceneRetrieval exactly as in the demo and never see the labels.
    Labels are used afterwards, only to score those predictions and to locate where the
    labelled product ended up (which is what the breakdown reports).
    """

    def __init__(
        self,
        image_downloader,
        image_encoder,
        catalog_index,
        garment_detector=None,
        recalls_at: Optional[List[int]] = None,
        query_cache: Optional[Path] = None,
        reuse_queries: bool = False,
    ):
        self.image_downloader = image_downloader
        self.image_encoder = image_encoder
        self.catalog_index = catalog_index
        self.garment_detector = garment_detector
        self.recalls_at = recalls_at or [1, 5, 10]
        self.evaluator = RetrievalEvaluator(self.recalls_at)
        self.query_cache = Path(query_cache) if query_cache else None
        self.reuse_queries = reuse_queries
        self.breakdown = []

        self.coverage_stats = {
            "total_validation_pairs": 0,
            "unique_scenes": 0,
            "scene_images_accessible": 0,
            "scene_images_unavailable": 0,
            "unique_ground_truth_products": 0,
            "gt_products_with_embeddings": 0,
            "eligible_scenes_evaluated": 0,
            "scenes_skipped_no_image": 0,
            "scenes_skipped_no_indexed_gt": 0,
        }

    def _scene_queries(self, retrieval, cache: dict, scene_id: str, image_path) -> Optional[dict]:
        """Detected-item queries for a scene; cached because detection and encoding dominate run time."""
        if scene_id in cache:
            return cache[scene_id]
        from PIL import Image
        try:
            scene_image = Image.open(image_path).convert("RGB")
        except Exception:
            return None
        built = retrieval.build_queries(scene_image)
        for q in built["queries"]:
            q["crop_image"] = None  # embeddings and attributes are enough for matching
        cache[scene_id] = built
        return built

    def _locate(self, index, garments: List[dict], gt_id: str) -> dict:
        """Where the labelled product sits for one scene (post-hoc diagnosis, not prediction)."""
        row = index._row_of[gt_id]
        gt_cat = str(index._categories[row])
        gt_alt = str(index._alt_categories[row])
        out = {
            "gt_category": gt_cat,
            "category_detected": False,
            "rank_in_category": None,
            "rank_after_rerank": None,
            "rank_overall": None,
        }
        best_overall = None
        for g in garments:
            emb = g.get("embedding")
            if emb is None:
                continue
            sims = index.ranking_scores(emb)
            best_overall = sims if best_overall is None else np.maximum(best_overall, sims)
            pool = index.CATEGORY_POOL.get(g["garment_label"], ([g["garment_label"]], []))[0]
            if g["garment_label"] == "Full Outfit / Look" or not (gt_cat in pool or (gt_alt and gt_alt in pool)):
                continue
            out["category_detected"] = True
            members = np.isin(index._categories, pool) | np.isin(index._alt_categories, pool)
            rank = int((sims[members] > sims[row]).sum()) + 1
            if out["rank_in_category"] is None or rank < out["rank_in_category"]:
                out["rank_in_category"] = rank
            recs = [r[0] for r in g["recommendations"]]
            if gt_id in recs:
                rr = recs.index(gt_id) + 1
                if out["rank_after_rerank"] is None or rr < out["rank_after_rerank"]:
                    out["rank_after_rerank"] = rr
        if best_overall is not None:
            out["rank_overall"] = int((best_overall > best_overall[row]).sum()) + 1
        return out

    def evaluate(self, validation_pairs: List[Dict], scene_to_products: Dict[str, List[str]]) -> Dict:
        """Run the demo pipeline on every validation scene and score it."""
        import pickle
        from src.retrieval import SceneRetrieval, CANDIDATE_POOL_SIZE

        index = self.catalog_index
        retrieval = SceneRetrieval(self.image_encoder, index, self.garment_detector)
        index.similarities(index.embeddings[0])  # prepares category arrays

        self.coverage_stats["total_validation_pairs"] = len(validation_pairs)
        self.coverage_stats["unique_scenes"] = len(scene_to_products)
        all_gt = set(p for prods in scene_to_products.values() for p in prods)
        indexed = set(index.get_all_product_ids())
        self.coverage_stats["unique_ground_truth_products"] = len(all_gt)
        self.coverage_stats["gt_products_with_embeddings"] = len(all_gt & indexed)

        cache = {}
        if self.reuse_queries and self.query_cache and self.query_cache.exists():
            with open(self.query_cache, "rb") as fh:
                cache = pickle.load(fh)
            logger.info(f"Reusing cached scene queries for {len(cache)} scenes")
        cached_before = len(cache)

        demo_hits = []
        exact_fired = exact_correct = 0

        for scene_id, gt_prods in scene_to_products.items():
            scene_img_path = self.image_downloader.get_cached_image(scene_id)
            if scene_img_path is None or not scene_img_path.exists():
                self.coverage_stats["scene_images_unavailable"] += 1
                self.coverage_stats["scenes_skipped_no_image"] += 1
                continue
            self.coverage_stats["scene_images_accessible"] += 1

            built = self._scene_queries(retrieval, cache, scene_id, scene_img_path)
            if built is None:
                continue
            # Full reranked list per item; cross-item de-duplication is a display feature and is
            # switched off here so each item's ranking can be inspected on its own.
            garments = retrieval.match_queries(built, top_k_per_garment=CANDIDATE_POOL_SIZE, remove_duplicates=False)

            eligible_gt = set(p for p in gt_prods if p in indexed)
            row = {
                "scene_id": scene_id,
                "labelled_product": ";".join(gt_prods),
                "product_indexed": bool(eligible_gt),
                "detected_items": ";".join(g["item_name"] for g in garments),
                "gt_category": "",
                "category_detected": "",
                "rank_in_category": "",
                "rank_after_rerank": "",
                "rank_overall": "",
                "fused_rank": "",
            }
            if not eligible_gt:
                self.coverage_stats["scenes_skipped_no_indexed_gt"] += 1
                self.breakdown.append(row)
                continue

            # Fuse candidates across items: each product keeps its best rerank score.
            product_best = {}
            shown_top3, shown_top5 = set(), set()
            for g in garments:
                for pid, _sim, score in g["recommendations"]:
                    if pid not in product_best or score > product_best[pid]:
                        product_best[pid] = score
                recs = [r[0] for r in g["recommendations"]]
                sims = [r[1] for r in g["recommendations"]]
                if sims and sims[0] >= 0.93 and (sims[0] >= 0.985 or len(sims) == 1 or sims[0] - max(sims[1:]) >= 0.03):
                    exact_fired += 1
                    exact_correct += int(recs[0] in gt_prods)
                shown_top3.update(recs[:3])
                shown_top5.update(recs[:5])
            if not product_best:
                self.breakdown.append(row)
                continue
            demo_hits.append((bool(shown_top3 & eligible_gt), bool(shown_top5 & eligible_gt)))

            fused = sorted(product_best.items(), key=lambda x: x[1], reverse=True)
            result = self.evaluator.evaluate_scene(scene_id, eligible_gt, fused)

            located = [self._locate(index, garments, gt) for gt in eligible_gt]
            best = min(located, key=lambda d: (d["rank_in_category"] is None, d["rank_in_category"] or 0))
            row.update({k: ("" if v is None else v) for k, v in best.items()})
            row["fused_rank"] = result["first_hit_rank"] or ""
            self.breakdown.append(row)

        if self.query_cache and len(cache) != cached_before:
            self.query_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(self.query_cache, "wb") as fh:
                pickle.dump(cache, fh)

        self.coverage_stats["eligible_scenes_evaluated"] = len(self.evaluator.results)
        metrics = self.evaluator.compute_metrics()
        if demo_hits:
            # Share of scenes where the labelled product is among the per-item cards the demo shows
            metrics["demo_hit@3_per_garment"] = float(np.mean([h[0] for h in demo_hits]))
            metrics["demo_hit@5_per_garment"] = float(np.mean([h[1] for h in demo_hits]))
        metrics["exact_badges_fired"] = exact_fired
        metrics["exact_badges_correct"] = exact_correct
        metrics["index_size"] = index.get_size()
        metrics["breakdown"] = self.summarize_breakdown()
        self.evaluator.metrics = metrics
        return {"metrics": metrics, "coverage": self.coverage_stats}

    def summarize_breakdown(self) -> Dict:
        """Aggregate the per-scene breakdown into the stages where the labelled product is lost."""
        scored = [r for r in self.breakdown if r["product_indexed"] and r["gt_category"]]
        detected = [r for r in scored if r["category_detected"] is True]

        def share_within(rows, key, k):
            return round(sum(1 for r in rows if r[key] != "" and r[key] <= k) / max(1, len(scored)), 4)

        summary = {
            "scenes_total": len(self.breakdown),
            "labelled_product_indexed": len(scored),
            "correct_category_detected": len(detected),
            "category_missed": len(scored) - len(detected),
        }
        for k in (1, 5, 10, 20, 50):
            summary[f"in_category_top{k}"] = share_within(detected, "rank_in_category", k)
            summary[f"after_rerank_top{k}"] = share_within(detected, "rank_after_rerank", k)
            summary[f"overall_top{k}"] = share_within(scored, "rank_overall", k)
        ranks = sorted(r["rank_in_category"] for r in detected if r["rank_in_category"] != "")
        summary["median_rank_in_category"] = ranks[len(ranks) // 2] if ranks else None
        missed = {}
        for r in scored:
            if r["category_detected"] is not True:
                missed[r["gt_category"]] = missed.get(r["gt_category"], 0) + 1
        summary["category_missed_by_type"] = dict(sorted(missed.items(), key=lambda x: -x[1]))
        return summary

    def save_breakdown(self, csv_path: Path):
        """Write one row per validation scene."""
        if not self.breakdown:
            return
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.breakdown[0].keys()))
            writer.writeheader()
            writer.writerows(self.breakdown)

    def print_summary(self):
        """Print coverage, metrics and the stage-by-stage breakdown."""
        c = self.coverage_stats
        print("\n" + "=" * 60)
        print("EVALUATION & COVERAGE SUMMARY")
        print("=" * 60)
        print(f"Validation pairs / scenes:       {c['total_validation_pairs']} / {c['unique_scenes']}")
        print(f"Scene images available:          {c['scene_images_accessible']}")
        print(f"Labelled products indexed:       {c['gt_products_with_embeddings']} / {c['unique_ground_truth_products']}")
        print(f"Scenes skipped (product not in catalog): {c['scenes_skipped_no_indexed_gt']}")
        self.evaluator.print_metrics()
        m = self.evaluator.metrics
        print(f"Index size:             {m.get('index_size')}")
        print(f"Demo hit@3 per garment: {m.get('demo_hit@3_per_garment', 0):.4f}")
        print(f"Demo hit@5 per garment: {m.get('demo_hit@5_per_garment', 0):.4f}")
        print(f"Exact badges fired/correct: {m.get('exact_badges_fired')}/{m.get('exact_badges_correct')}")
        b = m.get("breakdown", {})
        if b:
            print("-" * 60)
            print("BREAKDOWN (share of scored scenes; where the labelled product is lost)")
            print(f"Scored scenes:                    {b['labelled_product_indexed']}")
            print(f"Correct category detected:        {b['correct_category_detected']}  (missed: {b['category_missed']})")
            print(f"Missed by product type:           {b['category_missed_by_type']}")
            print(f"Median rank within category:      {b['median_rank_in_category']}")
            print("Top-K          in-category   after-rerank   overall(no category)")
            for k in (1, 5, 10, 20, 50):
                print(f"  top {k:<3}        {b[f'in_category_top{k}']:.4f}        {b[f'after_rerank_top{k}']:.4f}         {b[f'overall_top{k}']:.4f}")
        print("=" * 60)
