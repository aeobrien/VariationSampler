"""Tests for training data import and parsing."""

import pytest

from src.data.import_samples import parse_filename, scan_pass_directory, SampleRecord


class TestParseFilename:

    def test_standard_format(self):
        """Should parse standard AD2 capture filenames."""
        result = parse_filename("Track1_Snare_v127_rr04.wav")
        assert result is not None
        assert result["track_name"] == "Track1"
        assert result["instrument"] == "Snare"
        assert result["velocity"] == 127
        assert result["rr_index"] == 4
        assert result["group_key"] == "Track1_Snare_v127"

    def test_two_digit_track(self):
        """Should handle multi-digit track numbers."""
        result = parse_filename("Track20_Kick_v025_rr10.wav")
        assert result is not None
        assert result["track_name"] == "Track20"
        assert result["velocity"] == 25
        assert result["rr_index"] == 10

    def test_space_in_track_name(self):
        """Should normalize 'Track 1' to 'Track1'."""
        result = parse_filename("Track 1_Snare_v076_rr01.wav")
        assert result is not None
        assert result["track_name"] == "Track1"

    def test_all_instruments(self):
        """Should parse all known instrument types."""
        for inst in ["Kick", "Snare", "CrossStick", "Rimshot", "HiHatClosed"]:
            result = parse_filename(f"Track1_{inst}_v127_rr01.wav")
            assert result is not None
            assert result["instrument"] == inst

    def test_unknown_instrument_returns_none(self):
        """Should return None for unknown instruments."""
        result = parse_filename("Track1_Cymbal_v127_rr01.wav")
        assert result is None

    def test_non_matching_format_returns_none(self):
        """Should return None for non-matching filenames."""
        assert parse_filename("random_file.wav") is None
        assert parse_filename("not_a_sample.txt") is None

    def test_group_key_format(self):
        """Group key should be Track_Instrument_vVVV."""
        result = parse_filename("Track5_CrossStick_v051_rr07.wav")
        assert result["group_key"] == "Track5_CrossStick_v051"


class TestScanPassDirectory:

    def test_scan_real_data(self):
        """Scan actual pass-01 directory (integration test)."""
        from pathlib import Path
        pass_dir = Path("training-data/regions/pass-01")
        if not pass_dir.exists():
            pytest.skip("Training data not available")

        sets = scan_pass_directory(pass_dir, pass_id="pass-01")
        assert len(sets) > 0
        # Every set should have at least one sample
        for rr_set in sets.values():
            assert len(rr_set.samples) >= 1

    def test_nonexistent_dir_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            scan_pass_directory(tmp_path / "nonexistent")

    def test_empty_dir(self, tmp_path):
        """Should return empty dict for directory with no WAVs."""
        sets = scan_pass_directory(tmp_path)
        assert len(sets) == 0
