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
