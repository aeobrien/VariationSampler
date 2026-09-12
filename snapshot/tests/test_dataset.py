"""Tests for CodegramPairDataset."""

import numpy as np
import pytest
import torch

from src.utils.audio import NQ, T_MAX
from src.data.dataset import CodegramPairDataset
from src.data.codegram_cache import cache_codegram


@pytest.fixture
def cached_pair(codegram_pair, tmp_path):
    """Create a cached codegram pair on disk, return paths."""
    cg_a, cg_b = codegram_pair
    path_a = tmp_path / "a.npy"
    path_b = tmp_path / "b.npy"
    cache_codegram(cg_a, path_a)
    cache_codegram(cg_b, path_b)
    return str(path_a), str(path_b)


class TestCodegramPairDataset:

    def test_length(self, cached_pair):
        """Dataset length should match number of pairs."""
        dataset = CodegramPairDataset([cached_pair, cached_pair, cached_pair])
        assert len(dataset) == 3

    def test_shapes(self, cached_pair):
        """Items should be (z_A, z_B) with shape [NQ, T_MAX]."""
        dataset = CodegramPairDataset([cached_pair])
        z_a, z_b = dataset[0]
        assert z_a.shape == (NQ, T_MAX)
        assert z_b.shape == (NQ, T_MAX)

    def test_dtype(self, cached_pair):
        """Codegrams should be long (int64) tensors."""
        dataset = CodegramPairDataset([cached_pair])
        z_a, z_b = dataset[0]
        assert z_a.dtype == torch.long
        assert z_b.dtype == torch.long

    def test_value_range(self, cached_pair):
        """Token values should be in [0, 1023]."""
        dataset = CodegramPairDataset([cached_pair])
        z_a, z_b = dataset[0]
        assert z_a.min() >= 0 and z_a.max() <= 1023
        assert z_b.min() >= 0 and z_b.max() <= 1023

    def test_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing codegram."""
        dataset = CodegramPairDataset([
            (str(tmp_path / "missing_a.npy"), str(tmp_path / "missing_b.npy"))
        ])
        with pytest.raises(FileNotFoundError):
            dataset[0]
