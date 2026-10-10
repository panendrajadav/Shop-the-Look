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

    def search_candidates(
        self,
        query_embedding: np.ndarray,
        target_category: Optional[str] = None,
        candidate_pool_size: int = 80,
    ) -> List[Tuple[str, float]]:
        """Retrieve candidate products for reranking with category-aware indexing.

        Prioritizes candidates matching target_category and compatible categories,
        ensuring queries for jackets or t-shirts are not crowded out by boots.
        """
        if self.faiss_index is None or len(self.product_ids) == 0:
            return []

        # If no target category or no metadata, fall back to global FAISS search
        if not target_category or not self.metadata:
            return self.search(query_embedding, top_k=candidate_pool_size)

        compatibility_map = {
            "Jacket / Outerwear": {
                "preferred": ["Jacket / Outerwear"],
                "compatible": ["Top / T-shirt"],
            },
            "Top / T-shirt": {
                "preferred": ["Top / T-shirt"],
                "compatible": ["Jacket / Outerwear"],
            },
            "Pants / Trousers": {
                "preferred": ["Pants / Trousers"],
                "compatible": ["Skirt"],
            },
            "Dress": {
                "preferred": ["Dress"],
                "compatible": ["Skirt"],
            },
            "Skirt": {
                "preferred": ["Skirt"],
                "compatible": ["Pants / Trousers", "Dress"],
            },
            "Footwear": {
                "preferred": ["Footwear"],
                "compatible": [],
            },
            "Bag / Handbag": {
                "preferred": ["Bag / Handbag"],
                "compatible": [],
            },
        }

        rules = compatibility_map.get(target_category, {"preferred": [target_category], "compatible": []})
        preferred_cats = set(rules.get("preferred", []))
        compatible_cats = set(rules.get("compatible", []))

        # Perform inner-product similarity search across the catalog
        if self.embeddings is not None:
            if query_embedding.ndim == 1:
                q_vec = query_embedding.astype(np.float32)
            else:
                q_vec = query_embedding[0].astype(np.float32)

            all_sims = self.embeddings @ q_vec

            preferred_items = []
            compatible_items = []
            other_items = []

            for idx, sim in enumerate(all_sims):
                pid = self.product_ids[idx]
                cat = self.metadata.get(pid, {}).get("category", "")
                if cat in preferred_cats:
                    preferred_items.append((pid, float(sim)))
                elif cat in compatible_cats:
                    compatible_items.append((pid, float(sim)))
                else:
                    other_items.append((pid, float(sim)))

            preferred_items.sort(key=lambda x: x[1], reverse=True)
            compatible_items.sort(key=lambda x: x[1], reverse=True)
            other_items.sort(key=lambda x: x[1], reverse=True)

            # Assemble candidate pool: prioritized preferred and compatible items
            target_quota = min(len(preferred_items), int(candidate_pool_size * 0.70))
            compat_quota = min(len(compatible_items), int(candidate_pool_size * 0.25))

            candidates = preferred_items[:target_quota] + compatible_items[:compat_quota]
            remaining = candidate_pool_size - len(candidates)
            if remaining > 0:
                extra = preferred_items[target_quota:] + compatible_items[compat_quota:]
                candidates.extend(extra[:remaining])
                remaining = candidate_pool_size - len(candidates)
                if remaining > 0:
                    candidates.extend(other_items[:remaining])

            return candidates

        return self.search(query_embedding, top_k=candidate_pool_size)

    def search_with_category(
        self,
        query_embedding: np.ndarray,
        target_category: Optional[str] = None,
        top_k: int = 5,
        strict_category: bool = False,
    ) -> List[Tuple[str, float]]:
        """Search products with category filtering and category-aware prioritization.

        Args:
            query_embedding: Query embedding vector
            target_category: Optional target garment category to filter/boost
            top_k: Number of final results
            strict_category: If True, only returns exact category matches if available

        Returns:
            List of (product_id, similarity) tuples
        """
        # Retrieve candidate pool using category-aware search
        candidate_k = min(len(self.product_ids), max(top_k * 8, 60))
        candidates = self.search_candidates(query_embedding, target_category=target_category, candidate_pool_size=candidate_k)

        if not target_category or not self.metadata:
            return candidates[:top_k]

        matched_candidates = []
        compatible_candidates = []
        other_candidates = []

        compatibility_map = {
            "Jacket / Outerwear": ["Jacket / Outerwear", "Top / T-shirt"],
            "Top / T-shirt": ["Top / T-shirt", "Jacket / Outerwear"],
            "Pants / Trousers": ["Pants / Trousers", "Skirt"],
            "Skirt": ["Skirt", "Pants / Trousers", "Dress"],
            "Dress": ["Dress", "Skirt"],
            "Footwear": ["Footwear"],
            "Bag / Handbag": ["Bag / Handbag"],
        }
        allowed = compatibility_map.get(target_category, [target_category])

        for pid, sim in candidates:
            item_meta = self.metadata.get(pid, {})
            cat = item_meta.get("category", "")

            if cat == target_category:
                matched_candidates.append((pid, sim))
            elif cat in allowed:
                compatible_candidates.append((pid, sim))
            else:
                other_candidates.append((pid, sim))

        if strict_category:
            if len(matched_candidates) < top_k and candidate_k < len(self.product_ids):
                # Full catalog exact match search
                all_sims = self.embeddings @ (query_embedding.flatten().astype(np.float32))
                exact_items = [
                    (self.product_ids[i], float(all_sims[i]))
                    for i in range(len(self.product_ids))
                    if self.metadata.get(self.product_ids[i], {}).get("category") == target_category
                ]
                exact_items.sort(key=lambda x: x[1], reverse=True)
                return exact_items[:top_k]
            return matched_candidates[:top_k]

        combined = matched_candidates + compatible_candidates
        if len(combined) < top_k:
            combined = combined + other_candidates

        return combined[:top_k]

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
