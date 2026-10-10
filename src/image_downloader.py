import logging
import hashlib
import json
import threading
from pathlib import Path
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


class ImageDownloader:
    """Downloads and caches images from Pinterest CDN with caching, fallback, and parallelism."""

    def __init__(self, cache_dir: Path, timeout: int = 8, retry_attempts: int = 3, max_workers: int = 16):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.max_workers = max_workers
        self._lock = threading.Lock()

        self.download_stats = {
            "successful": 0,
            "failed": 0,
            "cached": 0,
            "invalid": 0,
        }

        self.metadata_file = self.cache_dir / "download_metadata.json"
        self.metadata = self._load_metadata()

    def _get_session(self) -> requests.Session:
        """Create a resilient requests session."""
        session = requests.Session()
        retries = Retry(
            total=self.retry_attempts,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=self.max_workers, pool_maxsize=self.max_workers * 2)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _load_metadata(self) -> dict:
        """Load download metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        return {}

    def _save_metadata(self):
        """Save download metadata."""
        with self._lock:
            try:
                with open(self.metadata_file, "w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save metadata: {e}")

    def _get_cache_path(self, image_id: str) -> Path:
        """Get canonical cache file path for an image ID."""
        id_hash = hashlib.md5(image_id.encode()).hexdigest()[:8]
        cache_subdir = self.cache_dir / id_hash[:2] / id_hash[2:4]
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / f"{image_id}.jpg"

    def _convert_to_url(self, signature: str, scheme: str = "https") -> str:
        """Convert signature to image URL using assignment convention."""
        return f"{scheme}://i.pinimg.com/400x/{signature[0:2]}/{signature[2:4]}/{signature[4:6]}/{signature}.jpg"

    def _validate_image(self, image_data: bytes) -> bool:
        """Validate image data is a decodable image."""
        try:
            img = Image.open(BytesIO(image_data))
            img.verify()
            return True
        except Exception:
            return False

    def download_image(self, image_id: str, session: Optional[requests.Session] = None) -> Optional[Path]:
        """Download and cache an image with fallback and validation.

        Args:
            image_id: The signature/ID for the image
            session: Optional requests session

        Returns:
            Path to cached image file, or None if download failed
        """
        cache_path = self._get_cache_path(image_id)

        # 1. Check if already cached on disk
        if cache_path.exists() and cache_path.stat().st_size > 0:
            with self._lock:
                self.download_stats["cached"] += 1
                if image_id not in self.metadata:
                    self.metadata[image_id] = {"status": "success", "path": str(cache_path)}
            return cache_path

        # Also check alternative folder layout if previously cached under image_id[:2]/image_id[2:4]
        alt_path = self.cache_dir / image_id[:2] / image_id[2:4] / f"{image_id}.jpg"
        if alt_path.exists() and alt_path.stat().st_size > 0:
            with self._lock:
                self.download_stats["cached"] += 1
                self.metadata[image_id] = {"status": "success", "path": str(alt_path)}
            return alt_path

        # 2. Skip if previously recorded as permanently failed
        with self._lock:
            meta = self.metadata.get(image_id)
            if meta and meta.get("status") == "failed" and meta.get("reason") in ["404_not_found", "invalid_image"]:
                self.download_stats["failed"] += 1
                return None

        # 3. Download image data (HTTPS first, then HTTP fallback)
        http_client = session or requests
        image_data = None
        last_error = None

        for scheme in ["https", "http"]:
            url = self._convert_to_url(image_id, scheme=scheme)
            try:
                response = http_client.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    image_data = response.content
                    break
                elif response.status_code == 404:
                    last_error = "404_not_found"
                    break
                else:
                    last_error = f"http_{response.status_code}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"

        if image_data is None:
            with self._lock:
                self.metadata[image_id] = {"status": "failed", "reason": last_error or "download_failed"}
                self.download_stats["failed"] += 1
            return None

        # 4. Validate image bytes
        if not self._validate_image(image_data):
            with self._lock:
                self.metadata[image_id] = {"status": "failed", "reason": "invalid_image"}
                self.download_stats["invalid"] += 1
            return None

        # 5. Save to disk
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(image_data)
            with self._lock:
                self.metadata[image_id] = {"status": "success", "path": str(cache_path)}
                self.download_stats["successful"] += 1
            return cache_path
        except Exception as e:
            logger.warning(f"Could not save image {image_id}: {e}")
            with self._lock:
                self.metadata[image_id] = {"status": "failed", "reason": f"save_error: {e}"}
                self.download_stats["failed"] += 1
            return None

    def download_images_parallel(self, image_ids: List[str], max_workers: Optional[int] = None) -> Dict[str, Path]:
        """Download multiple images concurrently using a thread pool.

        Args:
            image_ids: List of image IDs to download
            max_workers: Number of worker threads (defaults to self.max_workers)

        Returns:
            Dict mapping image_id -> Path for successfully downloaded/cached images
        """
        workers = max_workers or self.max_workers
        session = self._get_session()
        downloaded = {}

        logger.info(f"Downloading {len(image_ids)} images with {workers} worker threads...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_id = {
                executor.submit(self.download_image, img_id, session): img_id
                for img_id in image_ids
            }
            count = 0
            for future in as_completed(future_to_id):
                img_id = future_to_id[future]
                count += 1
                try:
                    path = future.result()
                    if path:
                        downloaded[img_id] = path
                except Exception as e:
                    logger.debug(f"Exception downloading {img_id}: {e}")

                if count % 200 == 0 or count == len(image_ids):
                    logger.info(f"Download progress: {count}/{len(image_ids)} ({len(downloaded)} available)")

        self._save_metadata()
        return downloaded

    def get_cached_image(self, image_id: str) -> Optional[Path]:
        """Get cached image path if it exists on disk and is non-empty."""
        # 1. Check metadata
        if image_id in self.metadata and self.metadata[image_id].get("status") == "success":
            stored_path = self.metadata[image_id].get("path")
            if stored_path and Path(stored_path).exists() and Path(stored_path).stat().st_size > 0:
                return Path(stored_path)

        # 2. Check standard cache path
        cache_path = self._get_cache_path(image_id)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path

        # 3. Check alternative cache path
        alt_path = self.cache_dir / image_id[:2] / image_id[2:4] / f"{image_id}.jpg"
        if alt_path.exists() and alt_path.stat().st_size > 0:
            return alt_path

        return None

    def get_stats(self) -> dict:
        """Get download statistics."""
        with self._lock:
            return self.download_stats.copy()

    def save_stats(self):
        """Save metadata to disk."""
        self._save_metadata()
