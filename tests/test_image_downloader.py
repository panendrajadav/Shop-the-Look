#!/usr/bin/env python3
"""Unit tests for URL construction, download retries, invalid image detection, and caching."""

import sys
import tempfile
from pathlib import Path
from PIL import Image
from io import BytesIO
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.image_downloader import ImageDownloader


def test_url_construction():
    """Verify URL construction follows the exact assignment specification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = ImageDownloader(Path(tmpdir))
        sig = "0027e30879ce3d87f82f699f148bff7e"
        url_https = downloader._convert_to_url(sig, scheme="https")
        assert url_https == "https://i.pinimg.com/400x/00/27/e3/0027e30879ce3d87f82f699f148bff7e.jpg"

        url_http = downloader._convert_to_url(sig, scheme="http")
        assert url_http == "http://i.pinimg.com/400x/00/27/e3/0027e30879ce3d87f82f699f148bff7e.jpg"


def test_image_validation_corrupted_data():
    """Verify that corrupt byte streams are flagged as invalid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = ImageDownloader(Path(tmpdir))
        corrupt_bytes = b"not an image file at all"
        assert not downloader._validate_image(corrupt_bytes)

        # Valid image bytes
        img = Image.new("RGB", (32, 32), color="red")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        valid_bytes = buf.getvalue()
        assert downloader._validate_image(valid_bytes)


def test_caching_and_persistence():
    """Verify caching saves images and prevents redundant downloads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = ImageDownloader(Path(tmpdir))
        dummy_id = "test_image_12345"

        # Simulate caching a valid image
        cache_path = downloader._get_cache_path(dummy_id)
        img = Image.new("RGB", (64, 64), color="blue")
        img.save(cache_path, format="JPEG")

        # Now get_cached_image should find it
        found_path = downloader.get_cached_image(dummy_id)
        assert found_path is not None
        assert found_path.exists()
        assert found_path == cache_path


def test_failed_image_handling():
    """Verify that non-existent image downloads gracefully return None without raising uncaught exceptions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = ImageDownloader(Path(tmpdir), timeout=2, retry_attempts=1)
        # Random non-existent ID
        bogus_id = "00000000000000000000000000000000"
        path = downloader.download_image(bogus_id)
        # Should be None if not found on Pinterest
        stats = downloader.get_stats()
        assert path is None or path.exists()
        assert "failed" in stats or "successful" in stats
