import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import faiss

logger = logging.getLogger(__name__)


class ProductIndex:
    """Manages product embeddings and metadata using FAISS for efficient search."""

    def __init__(self, embeddings_file: Path, metadata_file: Path, index_file: Path):
        self.embeddings_file = embeddings_file
        self.metadata_file = metadata_file
        self.index_file = index_file
        self.embeddings = None
        self.metadata = None
        self.faiss_index = None
        self.product_ids = []

    def build_index(self, product_id_to_embedding: Dict[str, np.ndarray],
                    product_id_to_metadata: Dict[str, dict]) -> int:
        """Build the index from product embeddings and metadata.

        Args:
            product_id_to_embedding: Dict mapping product_id -> embedding vector
            product_id_to_metadata: Dict mapping product_id -> metadata dict

        Returns:
            Number of products indexed
        """
        logger.info(f"Building index for {len(product_id_to_embedding)} products...")

        # Get embedding dimension
        sample_embedding = next(iter(product_id_to_embedding.values()))
        embedding_dim = sample_embedding.shape[0]

        # Build embeddings matrix
        embeddings_list = []
        self.product_ids = []
        metadata_list = []

        for prod_id, embedding in product_id_to_embedding.items():
            if embedding is None or len(embedding) != embedding_dim:
                continue
            embeddings_list.append(embedding)
            self.product_ids.append(prod_id)
            metadata_list.append(product_id_to_metadata.get(prod_id, {}))

        if not embeddings_list:
            logger.warning("No valid embeddings to index")
            return 0

        self.embeddings = np.array(embeddings_list, dtype=np.float32)

        # Create FAISS index
        logger.info(f"Creating FAISS index for {len(self.embeddings)} embeddings...")
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        self.faiss_index.add(self.embeddings)

        # Store metadata
        self.metadata = {
            prod_id: meta for prod_id, meta in zip(self.product_ids, metadata_list)
        }

        logger.info(f"Index built with {len(self.embeddings)} products")
        return len(self.embeddings)

    def save(self):
        """Save index, embeddings, and metadata to disk."""
        logger.info("Saving index...")

        # Save embeddings
        np.save(self.embeddings_file, self.embeddings)

        # Save metadata
        with open(self.metadata_file, "w") as f:
            json.dump({
                "product_ids": self.product_ids,
                "metadata": self.metadata
            }, f, indent=2)

        # Save FAISS index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(self.index_file))

        logger.info("Index saved successfully")

    def load(self) -> bool:
        """Load index, embeddings, and metadata from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load embeddings
            if self.embeddings_file.exists():
                self.embeddings = np.load(self.embeddings_file)
            else:
                logger.warning(f"Embeddings file not found: {self.embeddings_file}")
                return False

            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                self.product_ids = data["product_ids"]
                self.metadata = data["metadata"]
            else:
                logger.warning(f"Metadata file not found: {self.metadata_file}")
                return False

            # Load FAISS index
            if self.index_file.exists():
                self.faiss_index = faiss.read_index(str(self.index_file))
            else:
                logger.warning(f"FAISS index file not found: {self.index_file}")
                # Try to rebuild from embeddings
                embedding_dim = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)
                self.faiss_index.add(self.embeddings)

            # Optional garment-level views of products photographed on a model
            self.views = {}
            views_file = self.embeddings_file.with_name("products_views.npz")
            if views_file.exists():
                data = np.load(views_file)
                for region in np.unique(data["regions"]):
                    mask = data["regions"] == region
                    rows = data["rows"][mask]
                    keep = rows < len(self.product_ids)
                    self.views[str(region)] = (rows[keep], data["embeddings"][mask][keep])

            logger.info(f"Index loaded successfully. {len(self.product_ids)} products available")
            return True

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar products.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (product_id, similarity_score) tuples
        """
        if self.faiss_index is None or len(self.product_ids) == 0:
            logger.warning("Index is empty or not loaded")
            return []

        # Ensure query embedding is normalized and correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Search
        distances, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.product_ids)))

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.product_ids):
                product_id = self.product_ids[idx]
                similarity = float(distance)  # IP distance is already the similarity
                results.append((product_id, similarity))

        return results

    # Query category -> catalog categories that may be returned for it
    CATEGORY_POOL = {
        "Jacket / Outerwear": (["Jacket / Outerwear"], []),
        "Top / T-shirt": (["Top / T-shirt"], []),
        "Pants / Trousers": (["Pants / Trousers"], []),
        "Skirt": (["Skirt"], []),
        "Dress": (["Dress"], []),
        "Footwear": (["Footwear"], []),
        "Bag / Handbag": (["Bag / Handbag"], []),
        "Accessory": (["Accessory"], []),
        "Full Outfit / Look": (["Dress", "Jacket / Outerwear", "Top / T-shirt", "Pants / Trousers", "Skirt"], []),
    }
    UPPER_CATEGORIES = ("Top / T-shirt", "Jacket / Outerwear")
    LOWER_CATEGORIES = ("Pants / Trousers", "Skirt")

    def _prepare_search(self):
        """Cache per-product categories and attach each garment view to products of the matching kind."""
        self._categories = np.array([self.metadata.get(pid, {}).get("category", "") for pid in self.product_ids])
        self._item_names = np.array([self.metadata.get(pid, {}).get("item_name", "") for pid in self.product_ids])
        # Runner-up category, used only when the product photo is ambiguous about what is sold
        self._alt_categories = np.array([
            m.get("category_alt", "") if m.get("category_alt_confidence", 0.0) >= 0.20 else ""
            for m in (self.metadata.get(pid, {}) for pid in self.product_ids)
        ])
        self._row_of = {pid: i for i, pid in enumerate(self.product_ids)}
        self.color_hists = None
        color_file = self.embeddings_file.with_name("products_color_hist.npy")
        if color_file.exists():
            hists = np.load(color_file)
            if len(hists) == len(self.product_ids):
                self.color_hists = hists
        self._view_sets = []
        for region, cats in (("upper", self.UPPER_CATEGORIES), ("lower", self.LOWER_CATEGORIES)):
            if region in getattr(self, "views", {}):
                rows, embs = self.views[region]
                keep = np.isin(self._categories[rows], cats)
                self._view_sets.append((rows[keep], embs[keep]))

        # Domain centring. Scene crops and catalog photos occupy different regions of the CLIP
        # space: every scene crop shares a "street photo" component and every catalog image a
        # "product photo" component. Removing each side's mean before comparing leaves the
        # garment-specific part, which is what should decide the ranking. Both means are
        # computed without labels (scripts/build_query_stats.py for the query side).
        def _unit(x):
            return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

        self._query_mean = None
        mean_file = self.embeddings_file.with_name("query_domain_mean.npy")
        if mean_file.exists():
            self._query_mean = np.load(mean_file).astype(np.float32)
            self._centred = _unit(self.embeddings - self.embeddings.mean(axis=0)).astype(np.float32)
            self._centred_views = [(rows, _unit(embs - embs.mean(axis=0)).astype(np.float32))
                                   for rows, embs in self._view_sets]

    def similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query to every product.

        A top or jacket photographed on a model is scored by the better of its whole image and
        its upper-body view; pants and skirts by the whole image and the lower-body view. This
        compares garment with garment instead of garment with whole outfit.
        """
        if getattr(self, "_categories", None) is None or len(self._categories) != len(self.product_ids):
            self._prepare_search()
        q_vec = query_embedding.reshape(-1).astype(np.float32)
        sims = self.embeddings @ q_vec
        for rows, embs in self._view_sets:
            np.maximum.at(sims, rows, embs @ q_vec)
        return sims

    def ranking_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        """Scores used to order products for a detected item: domain-centred cosine similarity.

        Falls back to plain cosine similarity when no query-domain mean has been built.
        """
        if getattr(self, "_categories", None) is None or len(self._categories) != len(self.product_ids):
            self._prepare_search()
        if self._query_mean is None:
            return self.similarities(query_embedding)
        q = query_embedding.reshape(-1).astype(np.float32) - self._query_mean
        q = q / (np.linalg.norm(q) + 1e-8)
        scores = self._centred @ q
        for rows, embs in self._centred_views:
            np.maximum.at(scores, rows, embs @ q)
        return scores

    def search_candidates(
        self,
        query_embedding: np.ndarray,
        target_category: Optional[str] = None,
        candidate_pool_size: int = 80,
        target_item: Optional[str] = None,
    ) -> List[Tuple[str, float, float]]:
        """Candidate products for one detected item, restricted to that item's category.

        Returns (product_id, cosine_similarity, ranking_score) ordered by ranking score.

        A shirt query returns shirts, a skirt query skirts. Products outside the category enter
        the pool only when they are near-duplicates of the query image (similarity >= 0.97), so
        an exact catalog image is never lost to a wrong category estimate.
        """
        if self.faiss_index is None or len(self.product_ids) == 0:
            return []
        if not target_category or not self.metadata or self.embeddings is None:
            return [(pid, sim, sim) for pid, sim in self.search(query_embedding, top_k=candidate_pool_size)]

        sims = self.similarities(query_embedding)       # plain cosine: reported and used for duplicates
        scores = self.ranking_scores(query_embedding)   # domain-centred: used for ordering
        preferred, _ = self.CATEGORY_POOL.get(target_category, ([target_category], []))
        in_category = np.isin(self._categories, preferred) | np.isin(self._alt_categories, preferred)
        if target_category == "Accessory" and target_item:
            same_item = in_category & (self._item_names == target_item)
            if same_item.sum() >= 3:
                in_category = same_item

        order = np.argsort(-scores)
        picked = [int(i) for i in order if in_category[i]][:candidate_pool_size]
        chosen = set(picked)
        for i in np.argsort(-sims)[:10]:
            if sims[i] < 0.97:
                break
            if int(i) not in chosen:
                picked.append(int(i))
        if len(picked) < 3:  # nothing of this category in the catalog: fall back to closest overall
            picked = [int(i) for i in order[:candidate_pool_size]]
        picked.sort(key=lambda i: -scores[i])
        return [(self.product_ids[i], float(sims[i]), float(scores[i])) for i in picked]

    def color_similarity(self, product_id: str, query_hist: Optional[np.ndarray]) -> Optional[float]:
        """Histogram intersection (0..1) between a query crop and a product's garment colors."""
        if query_hist is None or getattr(self, "color_hists", None) is None:
            return None
        row = self._row_of.get(product_id)
        if row is None:
            return None
        return float(np.minimum(self.color_hists[row], query_hist).sum())

    def get_metadata(self, product_id: str) -> Optional[dict]:
        """Get metadata for a product."""
        if self.metadata is None:
            return None
        return self.metadata.get(product_id)

    def get_size(self) -> int:
        """Get number of products in index."""
        return len(self.product_ids) if self.product_ids else 0

    def get_all_product_ids(self) -> List[str]:
        """Get list of all product IDs in index."""
        return self.product_ids.copy()
