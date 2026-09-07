"""Tests for Claude API integration."""

import json

import pytest

from src.automation.claude_loop import (
    build_system_prompt,
    build_iteration_prompt,
    parse_config_response,
    validate_config_update,
)


ALLOWED_PARAMS = [
    "mask_p_attack",
    "mask_p_tail",
    "temperature",
    "top_p",
    "k_candidates",
    "learning_rate",
    "batch_size",
    "attack_frames",
    "editable_codebooks",
    "acceptance_band_low",
    "acceptance_band_high",
]

STEP_SIZE_LIMITS = {
    "mask_p_tail": 0.05,
    "mask_p_attack": 0.02,
    "temperature": 0.2,
    "top_p": 0.1,
    "learning_rate_factor": 2.0,
    "k_candidates": 4,
}


class TestBuildSystemPrompt:

    def test_includes_all_allowed_params(self):
        """System prompt should list all allowed parameters."""
        prompt = build_system_prompt(ALLOWED_PARAMS, STEP_SIZE_LIMITS)
        for param in ALLOWED_PARAMS:
            assert param in prompt

    def test_includes_step_limits(self):
        """System prompt should include step size limits."""
        prompt = build_system_prompt(ALLOWED_PARAMS, STEP_SIZE_LIMITS)
        assert "mask_p_tail" in prompt
        assert "0.05" in prompt

    def test_requires_reasoning(self):
        """System prompt should mention reasoning requirement."""
        prompt = build_system_prompt(ALLOWED_PARAMS, STEP_SIZE_LIMITS)
        assert "reasoning" in prompt.lower()

    def test_returns_string(self):
        """Should return a non-empty string."""
        prompt = build_system_prompt(ALLOWED_PARAMS, STEP_SIZE_LIMITS)
        assert isinstance(prompt, str)
        assert len(prompt) > 100


class TestBuildIterationPrompt:

    def test_includes_metrics(self):
        """Prompt should include metric values."""
        report = {
            "iteration_id": 3,
            "acceptance_rate": 0.75,
            "metrics": {
                "mrstft": {"mean": 0.45, "p50": 0.44, "p95": 0.6},
            },
            "trends": {"mrstft": "improving"},
            "config": {"sampling": {"temperature": 0.9}},
        }
        prompt = build_iteration_prompt(report)
        assert "0.45" in prompt
        assert "mrstft" in prompt

    def test_includes_listening_notes(self):
        """Prompt should include listening notes when provided."""
        report = {
            "iteration_id": 1,
            "metrics": {},
            "config": {},
        }
        prompt = build_iteration_prompt(report, listening_notes="Sounds too metallic")
        assert "Sounds too metallic" in prompt

    def test_includes_previous_reports(self):
        """Prompt should include abbreviated previous iteration data."""
        current = {
            "iteration_id": 3,
            "metrics": {"mrstft": {"mean": 0.4, "p50": 0.39, "p95": 0.5}},
            "config": {},
            "acceptance_rate": 0.8,
        }
        previous = [
            {
                "iteration_id": 1,
                "acceptance_rate": 0.6,
                "metrics": {"mrstft": {"mean": 0.5}},
            },
            {
                "iteration_id": 2,
                "acceptance_rate": 0.7,
                "metrics": {"mrstft": {"mean": 0.45}},
            },
        ]
        prompt = build_iteration_prompt(current, previous_reports=previous)
        assert "Iter 1" in prompt
        assert "Iter 2" in prompt


class TestParseConfigResponse:

    def test_parses_valid_json(self):
        """Should parse a valid JSON response."""
        response = json.dumps({
            "reasoning": "Increasing p_tail for more variation.",
            "config": {"mask_p_tail": 0.12},
        })
        config, reasoning = parse_config_response(response)
        assert config == {"mask_p_tail": 0.12}
        assert "p_tail" in reasoning

    def test_parses_json_in_code_block(self):
        """Should extract JSON from markdown code blocks."""
        response = """Here's my analysis:

```json
{
  "reasoning": "Temperature is too high, causing artifacts.",
  "config": {"temperature": 0.8}
}
```"""
        config, reasoning = parse_config_response(response)
        assert config == {"temperature": 0.8}
        assert "Temperature" in reasoning

    def test_rejects_missing_reasoning(self):
        """Should reject responses without reasoning."""
        response = json.dumps({"config": {"temperature": 0.8}})
        with pytest.raises(ValueError, match="reasoning"):
            parse_config_response(response)

    def test_rejects_empty_reasoning(self):
        """Should reject responses with empty reasoning."""
        response = json.dumps({"reasoning": "", "config": {}})
        with pytest.raises(ValueError, match="reasoning"):
            parse_config_response(response)

    def test_rejects_invalid_json(self):
        """Should reject unparseable responses."""
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_config_response("this is not json at all {{{")

    def test_accepts_empty_config(self):
        """Should accept empty config with reasoning."""
        response = json.dumps({
            "reasoning": "Metrics look good, no changes needed.",
            "config": {},
        })
        config, reasoning = parse_config_response(response)
        assert config == {}
        assert len(reasoning) > 0


class TestValidateConfigUpdate:

    def test_valid_update_passes(self):
        """A valid update should pass validation."""
        old_config = {
            "masking": {"p_tail": 0.08, "p_attack": 0.02},
            "sampling": {"temperature": 0.9},
        }
        new_config = {"mask_p_tail": 0.10}
        valid, errors = validate_config_update(
            new_config, old_config, ALLOWED_PARAMS, STEP_SIZE_LIMITS,
        )
        assert valid
        assert errors == []

    def test_rejects_unknown_keys(self):
        """Should reject parameters not in allowed list."""
        new_config = {"secret_param": 42}
        valid, errors = validate_config_update(
            new_config, {}, ALLOWED_PARAMS, STEP_SIZE_LIMITS,
        )
        assert not valid
        assert any("secret_param" in e for e in errors)

    def test_rejects_oversized_step(self):
        """Should reject changes exceeding step size limits."""
        old_config = {
            "masking": {"p_tail": 0.08},
        }
        # Step of 0.10 exceeds limit of 0.05
        new_config = {"mask_p_tail": 0.18}
        valid, errors = validate_config_update(
            new_config, old_config, ALLOWED_PARAMS, STEP_SIZE_LIMITS,
        )
        assert not valid
        assert any("step" in e.lower() or "exceeds" in e.lower() for e in errors)

    def test_accepts_within_step_limit(self):
        """Should accept changes within step size limits."""
        old_config = {
            "masking": {"p_tail": 0.08},
        }
        new_config = {"mask_p_tail": 0.12}  # Step of 0.04, limit is 0.05
        valid, errors = validate_config_update(
            new_config, old_config, ALLOWED_PARAMS, STEP_SIZE_LIMITS,
        )
        assert valid

    def test_empty_update_valid(self):
        """Empty config update should be valid."""
        valid, errors = validate_config_update(
            {}, {}, ALLOWED_PARAMS, STEP_SIZE_LIMITS,
        )
        assert valid
        assert errors == []

    def test_multiple_errors(self):
        """Should report all errors, not just the first."""
        new_config = {
            "unknown_param": 1,
            "mask_p_tail": 0.99,  # Way too big a step
        }
        old_config = {"masking": {"p_tail": 0.08}}
        valid, errors = validate_config_update(
            new_config, old_config, ALLOWED_PARAMS, STEP_SIZE_LIMITS,
        )
        assert not valid
        assert len(errors) >= 2
