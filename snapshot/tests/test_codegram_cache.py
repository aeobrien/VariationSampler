"""Tests for codegram caching (no DAC model needed)."""

import numpy as np
import pytest

from src.utils.audio import NQ, T_MAX
from src.data.codegram_cache import cache_codegram, load_codegram


class TestCacheAndLoad:

    def test_roundtrip(self, codegram, tmp_path):
        """Save and load should produce identical codegram."""
        path = tmp_path / "test.npy"
        cache_codegram(codegram, path)
        loaded = load_codegram(path)

        np.testing.assert_array_equal(codegram, loaded)

    def test_file_created(self, codegram, tmp_path):
        """Cache should create a .npy file."""
        path = tmp_path / "subdir" / "test.npy"
        cache_codegram(codegram, path)
        assert path.exists()

    def test_shape_preserved(self, codegram, tmp_path):
        """Shape should be preserved through save/load."""
        path = tmp_path / "test.npy"
        cache_codegram(codegram, path)
        loaded = load_codegram(path)
        assert loaded.shape == (NQ, T_MAX)

    def test_dtype_preserved(self, codegram, tmp_path):
        """Dtype should be preserved."""
        path = tmp_path / "test.npy"
        cache_codegram(codegram, path)
        loaded = load_codegram(path)
        assert loaded.dtype == codegram.dtype


class TestLoadErrors:

    def test_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Codegram file not found"):
            load_codegram(tmp_path / "nonexistent.npy")
