import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import open_clip

logger = logging.getLogger(__name__)


class CLIPImageEncoder:
    """Encodes images using OpenCLIP."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device(device)
        self.model = None
        self.preprocess = None
        self.embedding_dim = None
        self._load_model()

    def _load_model(self):
        """Load the CLIP model and preprocessing pipeline."""
        logger.info(f"Loading {self.model_name} model from {self.pretrained}...")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self.model.eval()

            # Get embedding dimension
            with torch.no_grad():
                # Create a dummy image to determine output size
                dummy_img = Image.new("RGB", (224, 224))
                dummy_input = self.preprocess(dummy_img).unsqueeze(0).to(self.device)
                dummy_features = self.model.encode_image(dummy_input)
                self.embedding_dim = dummy_features.shape[1]

            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def encode_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Encode a single image to embedding vector.

        Args:
            image_path: Path to the image file

        Returns:
            Normalized embedding vector (1D numpy array), or None if encoding fails
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.debug(f"Could not load image {image_path}: {e}")
            return None

        try:
            with torch.no_grad():
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_input)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0].astype(np.float32)
        except Exception as e:
            logger.debug(f"Error encoding image {image_path}: {e}")
            return None

    def encode_batch(self, image_paths: list, batch_size: int = 32) -> dict:
        """Encode a batch of images.

        Args:
            image_paths: List of Path objects
            batch_size: Batch size for processing

        Returns:
            Dict mapping image_path -> embedding (or None if failed)
        """
        embeddings = {}

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            valid_paths = []
            images = []

            # Load and preprocess images
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    images.append(self.preprocess(image))
                    valid_paths.append(path)
                except Exception as e:
                    embeddings[path] = None
                    logger.debug(f"Could not load image {path}: {e}")

            if not images:
                continue

            # Encode batch
            try:
                with torch.no_grad():
                    batch_tensor = torch.stack(images).to(self.device)
                    batch_features = self.model.encode_image(batch_tensor)
                    # Normalize
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

                # Store results
                for path, embedding in zip(valid_paths, batch_features.cpu().numpy()):
                    embeddings[path] = embedding.astype(np.float32)
            except Exception as e:
                logger.warning(f"Error encoding batch: {e}")
                for path in valid_paths:
                    embeddings[path] = None

        return embeddings

    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dim
