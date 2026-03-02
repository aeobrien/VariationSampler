"""Claude API integration for the automation loop.

Sends iteration reports to Claude and receives config updates.
Claude may only change allowed hyperparameters within step size limits.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_system_prompt(
    allowed_params: list[str],
    step_size_limits: dict[str, float],
) -> str:
    """Build the system prompt for Claude's role in the automation loop.

    Args:
        allowed_params: List of config keys Claude may change.
        step_size_limits: Dict of param_name -> max absolute step size.

    Returns:
        System prompt string.
    """
    param_list = "\n".join(f"  - {p}" for p in sorted(allowed_params))
    step_list = "\n".join(
        f"  - {k}: max change ±{v}" for k, v in sorted(step_size_limits.items())
    )

    return f"""You are a hyperparameter tuning assistant for the VariationSampler project.
Your role is to analyze evaluation metrics from drum sample variation generation
and suggest config updates to improve output quality.

## What you may change

You may ONLY change these parameters:
{param_list}

## Step size limits

Per-iteration maximum changes:
{step_list}

For parameters without explicit step size limits, use conservative changes.

## What you may NOT change

- Source code
- Model architecture
- Loss function
- Evaluation metric implementations
- Any parameter not in the allowed list above

## Response format

You MUST respond with a JSON object containing exactly two fields:
1. "reasoning": A string explaining your analysis and why you're making these changes.
2. "config": A dict containing ONLY the parameters you want to change (not the full config).

Example response:
```json
{{
  "reasoning": "MR-STFT distance is below the target band, suggesting variations are too similar to the source. Increasing mask_p_tail should produce more variation.",
  "config": {{
    "mask_p_tail": 0.12
  }}
}}
```

If you believe no changes are needed, return an empty config dict with reasoning explaining why.
The "reasoning" field is REQUIRED. Responses without reasoning will be rejected."""


def build_iteration_prompt(
    iteration_report: dict[str, Any],
    previous_reports: list[dict[str, Any]] | None = None,
    listening_notes: str | None = None,
) -> str:
    """Build the user prompt with iteration report data.

    Args:
        iteration_report: Current iteration's report dict.
        previous_reports: Optional list of previous iteration reports for context.
        listening_notes: Optional listening notes from project owner.

    Returns:
        User prompt string.
    """
    parts = []

    parts.append("## Current Iteration Report")
    parts.append(f"Iteration: {iteration_report.get('iteration_id', 'N/A')}")
    parts.append(f"Acceptance rate: {iteration_report.get('acceptance_rate', 'N/A')}")
    parts.append("")

    # Current metrics
    parts.append("### Metrics")
    metrics = iteration_report.get("metrics", {})
    for name, summary in sorted(metrics.items()):
        if isinstance(summary, dict):
            parts.append(
                f"- {name}: mean={summary.get('mean', 'N/A'):.4f}, "
                f"p50={summary.get('p50', 'N/A'):.4f}, "
                f"p95={summary.get('p95', 'N/A'):.4f}"
            )
    parts.append("")

    # Trends
    trends = iteration_report.get("trends", {})
    if trends:
        parts.append("### Trends vs Previous")
        for name, trend in sorted(trends.items()):
            parts.append(f"- {name}: {trend}")
        parts.append("")

    # Baseline comparison
    baseline_comp = iteration_report.get("baseline_comparison", {})
    if baseline_comp:
        parts.append("### Baseline Comparison")
        for name, comp in sorted(baseline_comp.items()):
            in_band = "IN BAND" if comp.get("in_band") else "OUT OF BAND"
            parts.append(
                f"- {name}: value={comp.get('value', 'N/A'):.4f}, "
                f"baseline_median={comp.get('baseline_median', 'N/A'):.4f}, "
                f"status={in_band}"
            )
        parts.append("")

    # Current config
    parts.append("### Current Config")
    config = iteration_report.get("config", {})
    parts.append(f"```json\n{json.dumps(config, indent=2)}\n```")
    parts.append("")

    # Previous iteration history (abbreviated)
    if previous_reports:
        parts.append("### Previous Iterations (abbreviated)")
        for prev in previous_reports[-3:]:  # Last 3 iterations
            prev_id = prev.get("iteration_id", "?")
            prev_acceptance = prev.get("acceptance_rate", "N/A")
            prev_metrics = prev.get("metrics", {})
            mrstft_mean = prev_metrics.get("mrstft", {}).get("mean", "N/A")
            parts.append(
                f"- Iter {prev_id}: acceptance={prev_acceptance}, mrstft_mean={mrstft_mean}"
            )
        parts.append("")

    # Listening notes
    if listening_notes:
        parts.append("### Listening Notes from Project Owner")
        parts.append(listening_notes)
        parts.append("")

    parts.append("Based on this data, what config changes do you recommend?")

    return "\n".join(parts)


def call_claude(
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    log_dir: str | Path | None = None,
) -> str:
    """Call the Anthropic API with the given prompts.

    Args:
        system_prompt: System prompt string.
        user_prompt: User prompt string.
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        model: Model ID to use.
        log_dir: Optional directory to log prompts and responses.

    Returns:
        Raw response text from Claude.

    Raises:
        RuntimeError: If API call fails.
    """
    import anthropic

    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key provided. Set ANTHROPIC_API_KEY environment variable "
            "or pass api_key parameter."
        )

    client = anthropic.Anthropic(api_key=api_key)

    logger.info("Calling Claude API (model=%s)...", model)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    response_text = response.content[0].text

    # Log prompt and response
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"claude-{timestamp}.json"
        log_data = {
            "timestamp": timestamp,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response_text,
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        logger.info("Logged Claude API call: %s", log_path)

    return response_text


def parse_config_response(response_text: str) -> tuple[dict[str, Any], str]:
    """Extract config JSON and reasoning from Claude's response.

    Args:
        response_text: Raw response text from Claude.

    Returns:
        Tuple of (config_dict, reasoning_string).

    Raises:
        ValueError: If response cannot be parsed or is missing reasoning.
    """
    # Try to extract JSON from markdown code block first
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)

    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try parsing the entire response as JSON
        json_str = response_text.strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")

    reasoning = parsed.get("reasoning")
    if not reasoning or not isinstance(reasoning, str) or len(reasoning.strip()) == 0:
        raise ValueError("Response missing required 'reasoning' field")

    config = parsed.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"'config' must be a dict, got {type(config).__name__}")

    return config, reasoning


def validate_config_update(
    new_config: dict[str, Any],
    old_config: dict[str, Any],
    allowed_params: list[str],
    step_size_limits: dict[str, float],
) -> tuple[bool, list[str]]:
    """Validate a proposed config update from Claude.

    Checks that only allowed keys were changed and step sizes are within limits.

    Args:
        new_config: Proposed config changes (partial dict, not full config).
        old_config: Current full config dict.
        allowed_params: List of allowed parameter names.
        step_size_limits: Dict of param_name -> max absolute step size.

    Returns:
        Tuple of (valid, list_of_errors).
    """
    errors: list[str] = []

    # Check for unknown keys
    for key in new_config:
        if key not in allowed_params:
            errors.append(f"Parameter '{key}' is not in the allowed list")

    # Check step sizes
    for key, new_value in new_config.items():
        if key not in allowed_params:
            continue

        old_value = _find_param_in_config(key, old_config)
        if old_value is None:
            # New param being set — check if it's a reasonable value
            continue

        if key in step_size_limits:
            limit = step_size_limits[key]

            # Handle multiplicative limits (e.g., learning_rate_factor)
            if key == "learning_rate_factor":
                if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                    if old_value > 0 and new_value > 0:
                        ratio = max(new_value / old_value, old_value / new_value)
                        if ratio > limit:
                            errors.append(
                                f"'{key}' change ratio {ratio:.2f} exceeds limit {limit}"
                            )
            elif isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                step = abs(new_value - old_value)
                if step > limit:
                    errors.append(
                        f"'{key}' step {step} exceeds limit {limit} "
                        f"(old={old_value}, new={new_value})"
                    )

    valid = len(errors) == 0
    if not valid:
        logger.warning("Config validation failed: %s", errors)
    return valid, errors


def _find_param_in_config(key: str, config: dict) -> Any:
    """Find a parameter value in a potentially nested config.

    Searches both flat keys and common nesting patterns.
    """
    # Direct lookup
    if key in config:
        return config[key]

    # Common nesting patterns
    section_map = {
        "mask_p_attack": ("masking", "p_attack"),
        "mask_p_tail": ("masking", "p_tail"),
        "temperature": ("sampling", "temperature"),
        "top_p": ("sampling", "top_p"),
        "k_candidates": ("sampling", "k_candidates"),
        "learning_rate": ("training", "learning_rate"),
        "batch_size": ("training", "batch_size"),
        "attack_frames": ("masking", "attack_frames"),
        "editable_codebooks": ("model", "edit_codebooks"),
        "acceptance_band_low": ("acceptance", "mrstft_band",),
        "acceptance_band_high": ("acceptance", "mrstft_band",),
    }

    if key in section_map:
        path = section_map[key]
        value = config
        for part in path:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value

    return None
