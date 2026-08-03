"""Tests for train/dev/test split generation."""

import pytest

from src.data.splits import (
    GroupKey,
    SplitManifest,
    generate_splits,
    verify_no_leakage,
    generate_training_pairs,
)


class TestGenerateSplits:

    def test_disjoint_splits(self, sample_group_keys):
        """No group key should appear in more than one split."""
        manifest = generate_splits(
            sample_group_keys, test_libraries=["libC"], dev_fraction=0.15,
        )
        train_keys = set(manifest.train.keys())
        dev_keys = set(manifest.dev.keys())
        test_keys = set(manifest.test.keys())

        assert len(train_keys & dev_keys) == 0
        assert len(train_keys & test_keys) == 0
        assert len(dev_keys & test_keys) == 0

    def test_held_out_libraries_in_test(self, sample_group_keys):
        """All groups from test_libraries should be in test split."""
        manifest = generate_splits(
            sample_group_keys, test_libraries=["libC"],
        )
        for key in manifest.test.keys():
            assert key.startswith("libC/")

        # libC groups should not be in train or dev
        for key in manifest.train.keys():
            assert not key.startswith("libC/")
        for key in manifest.dev.keys():
            assert not key.startswith("libC/")

    def test_all_keys_covered(self, sample_group_keys):
        """Every input group should appear in exactly one split."""
        manifest = generate_splits(
            sample_group_keys, test_libraries=["libC"],
        )
        all_output = set(manifest.train.keys()) | set(manifest.dev.keys()) | set(manifest.test.keys())
        assert all_output == set(sample_group_keys.keys())

    def test_verify_no_leakage_passes(self, sample_group_keys):
        """verify_no_leakage should return True for valid splits."""
        manifest = generate_splits(
            sample_group_keys, test_libraries=["libC"],
        )
        assert verify_no_leakage(manifest) is True

    def test_verify_no_leakage_detects_leak(self):
        """verify_no_leakage should raise on overlapping splits."""
        manifest = SplitManifest(
            train={"key1": ["a.npy"], "key2": ["b.npy"]},
            dev={"key2": ["b.npy"]},  # leaked!
            test={},
        )
        with pytest.raises(ValueError, match="Data leakage"):
            verify_no_leakage(manifest)


class TestManifestRoundtrip:

    def test_save_and_load(self, sample_group_keys, tmp_path):
        """Manifest should survive save/load roundtrip."""
        manifest = generate_splits(
            sample_group_keys, test_libraries=["libC"],
        )
        path = tmp_path / "manifest.json"
        manifest.save(path)

        loaded = SplitManifest.load(path)
        assert loaded.train == manifest.train
        assert loaded.dev == manifest.dev
        assert loaded.test == manifest.test


class TestGenerateTrainingPairs:

    def test_pair_count(self):
        """N files should yield N*(N-1) pairs."""
        files = ["a.npy", "b.npy", "c.npy", "d.npy", "e.npy"]
        pairs = generate_training_pairs(files)
        assert len(pairs) == 5 * 4  # N*(N-1) = 20

    def test_no_self_pairs(self):
        """No pair should have the same file as input and target."""
        files = ["a.npy", "b.npy", "c.npy"]
        pairs = generate_training_pairs(files)
        for a, b in pairs:
            assert a != b

    def test_single_file_no_pairs(self):
        """Single file should produce no pairs."""
        pairs = generate_training_pairs(["a.npy"])
        assert len(pairs) == 0

    def test_empty_input(self):
        """Empty input should produce no pairs."""
        pairs = generate_training_pairs([])
        assert len(pairs) == 0


class TestGroupKey:

    def test_str_format(self):
        key = GroupKey("libA", "kit1", "snare", "hit", "close")
        assert str(key) == "libA/kit1/snare/hit/close"

    def test_frozen(self):
        key = GroupKey("libA", "kit1", "snare", "hit")
        with pytest.raises(AttributeError):
            key.library_id = "libB"
