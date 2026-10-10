"""Tests for the evaluation pipeline module."""

import json

import numpy as np
import pytest

from src.automation.evaluation import build_dev_sample_list
from src.utils.instrument_families import infer_family, INSTRUMENT_FAMILIES


class TestBuildDevSampleList:

    def test_builds_from_manifest(self, tmp_path):
        """Should build sample list from manifest.json."""
        # Create mock directory structure
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        codegrams_dir = tmp_path / "codegrams" / "pass-02"
        processed_dir = tmp_path / "processed" / "pass-02"

        # Create mock codegram and audio files
        for group in ["Track1_Snare_v127", "Track1_Kick_v127"]:
            cg_dir = codegrams_dir / group
            cg_dir.mkdir(parents=True)
            np.save(str(cg_dir / "hit_01.npy"), np.zeros((9, 100), dtype=np.int32))

            audio_dir = processed_dir / group
            audio_dir.mkdir(parents=True)
            # Create a minimal WAV file
            import soundfile as sf
            audio = np.zeros((4410, 1), dtype=np.float32)
            sf.write(str(audio_dir / "hit_01.wav"), audio, 44100)

        # Create manifest
        manifest = {
            "dev": {
                "Track1_Snare_v127": [
                    "data/processed/pass-02/Track1_Snare_v127/hit_01.wav",
                ],
                "Track1_Kick_v127": [
                    "data/processed/pass-02/Track1_Kick_v127/hit_01.wav",
                ],
            },
            "train": {},
            "test": {},
        }
        with open(splits_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        samples = build_dev_sample_list(
            splits_dir=splits_dir,
            codegrams_dir=codegrams_dir,
            processed_dir=processed_dir,
        )
        assert len(samples) == 2
        for s in samples:
            assert "name" in s
            assert "codegram" in s
            assert "audio" in s

    def test_max_samples_limits(self, tmp_path):
        """Should respect max_samples parameter."""
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        codegrams_dir = tmp_path / "codegrams" / "pass-02"
        processed_dir = tmp_path / "processed" / "pass-02"

        groups = {}
        for i in range(5):
            group = f"Track{i}_Snare_v127"
            cg_dir = codegrams_dir / group
            cg_dir.mkdir(parents=True)
            np.save(str(cg_dir / "hit_01.npy"), np.zeros((9, 100), dtype=np.int32))

            audio_dir = processed_dir / group
            audio_dir.mkdir(parents=True)
            import soundfile as sf
            sf.write(str(audio_dir / "hit_01.wav"), np.zeros((4410, 1), dtype=np.float32), 44100)

            groups[group] = [f"data/processed/pass-02/{group}/hit_01.wav"]

        manifest = {"dev": groups, "train": {}, "test": {}}
        with open(splits_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        samples = build_dev_sample_list(
            splits_dir, codegrams_dir, processed_dir, max_samples=2,
        )
        assert len(samples) == 2

    def test_missing_manifest_raises(self, tmp_path):
        """Should raise if manifest.json is missing."""
        with pytest.raises(FileNotFoundError):
            build_dev_sample_list(
                tmp_path / "nonexistent",
                tmp_path / "cg",
                tmp_path / "proc",
            )

    def test_skips_missing_files(self, tmp_path):
        """Should skip entries where codegram or audio file doesn't exist."""
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()

        manifest = {
            "dev": {
                "missing_group": [
                    "data/processed/pass-02/missing_group/hit_01.wav",
                ],
            },
        }
        with open(splits_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        samples = build_dev_sample_list(
            splits_dir, tmp_path / "cg", tmp_path / "proc",
        )
        assert len(samples) == 0

    def test_includes_family_key(self, tmp_path):
        """Each sample dict should include a 'family' key."""
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        codegrams_dir = tmp_path / "codegrams" / "pass-02"
        processed_dir = tmp_path / "processed" / "pass-02"

        for group in ["Track1_Snare_v127", "Track1_Kick_v127"]:
            cg_dir = codegrams_dir / group
            cg_dir.mkdir(parents=True)
            np.save(str(cg_dir / "hit_01.npy"), np.zeros((9, 100), dtype=np.int32))
            audio_dir = processed_dir / group
            audio_dir.mkdir(parents=True)
            import soundfile as sf
            sf.write(str(audio_dir / "hit_01.wav"), np.zeros((4410, 1), dtype=np.float32), 44100)

        manifest = {
            "dev": {
                "Track1_Snare_v127": ["data/processed/pass-02/Track1_Snare_v127/hit_01.wav"],
                "Track1_Kick_v127": ["data/processed/pass-02/Track1_Kick_v127/hit_01.wav"],
            },
        }
        with open(splits_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        samples = build_dev_sample_list(splits_dir, codegrams_dir, processed_dir)
        assert len(samples) == 2
        families = {s["family"] for s in samples}
        assert families == {"Snare", "Kick"}

    def test_samples_per_family(self, tmp_path):
        """Should select N samples per family when samples_per_family is set."""
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        codegrams_dir = tmp_path / "codegrams" / "pass-02"
        processed_dir = tmp_path / "processed" / "pass-02"

        groups = {}
        # 3 snares + 3 kicks
        for i in range(3):
            for inst in ["Snare", "Kick"]:
                group = f"Track{i}_{inst}_v127"
                cg_dir = codegrams_dir / group
                cg_dir.mkdir(parents=True)
                np.save(str(cg_dir / "hit_01.npy"), np.zeros((9, 100), dtype=np.int32))
                audio_dir = processed_dir / group
                audio_dir.mkdir(parents=True)
                import soundfile as sf
                sf.write(str(audio_dir / "hit_01.wav"), np.zeros((4410, 1), dtype=np.float32), 44100)
                groups[group] = [f"data/processed/pass-02/{group}/hit_01.wav"]

        manifest = {"dev": groups, "train": {}, "test": {}}
        with open(splits_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        # Request 2 per family -> 4 total (2 Snare + 2 Kick)
        samples = build_dev_sample_list(
            splits_dir, codegrams_dir, processed_dir, samples_per_family=2,
        )
        assert len(samples) == 4
        from collections import Counter
        family_counts = Counter(s["family"] for s in samples)
        assert family_counts["Snare"] == 2
        assert family_counts["Kick"] == 2


class TestInferFamily:

    def test_known_instruments(self):
        """Should map known instrument names to families."""
        assert infer_family("Track1_Snare_v127") == "Snare"
        assert infer_family("Track2_SnareRim_v100") == "Snare"
        assert infer_family("Track1_HiHatClosed_v127") == "HiHat"
        assert infer_family("Track3_PedalHat_v025") == "HiHat"
        assert infer_family("Track1_Kick_v127") == "Kick"
        assert infer_family("Track1_Rimshot_v127") == "Rimshot"
        assert infer_family("Track1_CrossStick_v127") == "CrossStick"

    def test_unknown_instrument(self):
        """Should return None for unrecognised instrument names."""
        assert infer_family("Track1_Cowbell_v127") is None

    def test_invalid_format(self):
        """Should return None for non-matching group key format."""
        assert infer_family("not_a_valid_key") is None
        assert infer_family("") is None
