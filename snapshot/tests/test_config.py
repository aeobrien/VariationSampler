"""Tests for config loading and validation."""

import pytest

from src.utils.config import load_config, merge_configs, validate_config


class TestLoadConfig:

    def test_default_config_loads(self):
        """Default config should load without error."""
        config = load_config("configs/default.yaml")
        assert isinstance(config, dict)
        assert "model" in config
        assert "masking" in config
        assert "sampling" in config
        assert "training" in config

    def test_missing_file_raises(self):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestMergeConfigs:

    def test_override_value(self):
        """Override should replace base values."""
        base = {"model": {"d_model": 256, "n_layers": 4}}
        override = {"model": {"d_model": 512}}
        merged = merge_configs(base, override)
        assert merged["model"]["d_model"] == 512
        assert merged["model"]["n_layers"] == 4

    def test_add_new_key(self):
        """Override can add new keys."""
        base = {"a": 1}
        override = {"b": 2}
        merged = merge_configs(base, override)
        assert merged == {"a": 1, "b": 2}

    def test_deep_merge(self):
        """Nested dicts should merge recursively."""
        base = {"x": {"y": {"z": 1, "w": 2}}}
        override = {"x": {"y": {"z": 3}}}
        merged = merge_configs(base, override)
        assert merged["x"]["y"]["z"] == 3
        assert merged["x"]["y"]["w"] == 2

    def test_does_not_mutate_inputs(self):
        """Original dicts should not be modified."""
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        merge_configs(base, override)
        assert base["a"]["b"] == 1


class TestValidateConfig:

    def test_default_config_valid(self):
        """Default config should pass validation."""
        config = load_config("configs/default.yaml")
        errors = validate_config(config)
        assert errors == []

    def test_missing_section(self):
        """Should detect missing required sections."""
        config = {"model": {"d_model": 256, "n_layers": 4, "n_heads": 4}}
        errors = validate_config(config)
        assert any("masking" in e for e in errors)

    def test_out_of_range_value(self):
        """Should detect out-of-range values."""
        config = load_config("configs/default.yaml")
        config["sampling"]["temperature"] = 100.0  # way too high
        errors = validate_config(config)
        assert any("temperature" in e for e in errors)

    def test_wrong_type(self):
        """Should detect wrong types."""
        config = load_config("configs/default.yaml")
        config["model"]["d_model"] = "not_an_int"
        errors = validate_config(config)
        assert any("d_model" in e for e in errors)
