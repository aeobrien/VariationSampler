"""Batch summary report generator for aggregating automation loop results."""

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.machine_gun_proxy import render_machine_gun
from src.utils.audio import save_wav, SAMPLE_RATE

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _compute_config_diff(config_a: dict, config_b: dict, prefix: str = "") -> list[dict]:
    """Compute differences between two config dicts.

    Returns:
        List of {"key": dotted_key, "old": old_value, "new": new_value}.
    """
    diffs = []
    all_keys = set(list(config_a.keys()) + list(config_b.keys()))
    for key in sorted(all_keys):
        dotted = f"{prefix}.{key}" if prefix else key
        val_a = config_a.get(key)
        val_b = config_b.get(key)
        if isinstance(val_a, dict) and isinstance(val_b, dict):
            diffs.extend(_compute_config_diff(val_a, val_b, prefix=dotted))
        elif val_a != val_b:
            diffs.append({"key": dotted, "old": val_a, "new": val_b})
    return diffs


def _trend_arrow(values: list[float]) -> str:
    """Compute trend direction from a list of values across iterations."""
    if len(values) < 2:
        return "—"
    first_half = np.mean(values[:len(values) // 2])
    second_half = np.mean(values[len(values) // 2:])
    if first_half == 0:
        return "—"
    change = (second_half - first_half) / abs(first_half)
    if change < -0.05:
        return "↓"  # decreasing
    elif change > 0.05:
        return "↑"  # increasing
    return "→"  # stable


def generate_batch_summary(
    batch_id: str,
    iteration_reports: list[dict[str, Any]],
    listening_notes_path: str | Path | None = None,
    stop_reason: str = "iteration_cap",
) -> dict[str, Any]:
    """Aggregate iteration reports into a batch summary.

    Args:
        batch_id: Batch identifier string.
        iteration_reports: List of iteration report dicts (in order).
        listening_notes_path: Optional path to listening notes markdown file.
        stop_reason: Why the batch stopped (iteration_cap, stagnation, regression, error).

    Returns:
        Batch summary dict.
    """
    if not iteration_reports:
        return {
            "batch_id": batch_id,
            "n_iterations": 0,
            "stop_reason": stop_reason,
            "error": "No iteration reports provided",
        }

    start_time = iteration_reports[0].get("timestamp", "")
    end_time = iteration_reports[-1].get("timestamp", "")

    # Config trajectory: diffs between consecutive iterations
    config_trajectory = []
    for i in range(1, len(iteration_reports)):
        prev_config = iteration_reports[i - 1].get("config", {})
        curr_config = iteration_reports[i].get("config", {})
        diffs = _compute_config_diff(prev_config, curr_config)
        config_trajectory.append({
            "iteration": i,
            "changes": diffs,
        })

    # Metric trajectory: per-metric values across iterations
    all_metric_names = set()
    for report in iteration_reports:
        all_metric_names.update(report.get("metrics", {}).keys())

    metric_trajectory: dict[str, dict] = {}
    for metric_name in sorted(all_metric_names):
        values = []
        for report in iteration_reports:
            metric_summary = report.get("metrics", {}).get(metric_name, {})
            values.append(metric_summary.get("mean", None))
        # Filter out None values for trend
        valid_values = [v for v in values if v is not None]
        metric_trajectory[metric_name] = {
            "values": values,
            "trend": _trend_arrow(valid_values),
        }

    # Best/worst samples across entire batch (by composite score)
    all_samples: list[dict] = []
    for report in iteration_reports:
        iteration_id = report.get("iteration_id", -1)
        for sample in report.get("best_samples", []):
            all_samples.append({**sample, "iteration": iteration_id})
        for sample in report.get("worst_samples", []):
            all_samples.append({**sample, "iteration": iteration_id})

    all_samples.sort(key=lambda x: x.get("score", float("inf")))
    # Deduplicate by name, keeping best score
    seen_names: set[str] = set()
    unique_samples: list[dict] = []
    for s in all_samples:
        if s["name"] not in seen_names:
            seen_names.add(s["name"])
            unique_samples.append(s)

    best_5 = unique_samples[:5]
    worst_5 = unique_samples[-5:]

    # Claude diagnoses (if present in reports)
    claude_diagnoses = []
    for report in iteration_reports:
        if "claude_reasoning" in report:
            claude_diagnoses.append({
                "iteration": report["iteration_id"],
                "reasoning": report["claude_reasoning"],
            })

    # Recommended next actions from final iteration
    final_report = iteration_reports[-1]
    recommended_actions = final_report.get("recommended_actions", [])

    # Load listening notes if provided
    listening_notes = None
    if listening_notes_path:
        path = Path(listening_notes_path)
        if path.exists():
            listening_notes = path.read_text()

    # Per-family data from final iteration
    per_family = final_report.get("per_family", {})

    summary = {
        "batch_id": batch_id,
        "start_time": start_time,
        "end_time": end_time,
        "n_iterations": len(iteration_reports),
        "stop_reason": stop_reason,
        "git_commit": _get_git_commit(),
        "config_trajectory": config_trajectory,
        "metric_trajectory": metric_trajectory,
        "per_family": per_family,
        "best_samples": best_5,
        "worst_samples": worst_5,
        "claude_diagnoses": claude_diagnoses,
        "recommended_actions": recommended_actions,
        "listening_notes": listening_notes,
        "starting_config": iteration_reports[0].get("config", {}),
        "final_config": iteration_reports[-1].get("config", {}),
    }

    logger.info(
        "Generated batch summary '%s': %d iterations, stopped: %s",
        batch_id, len(iteration_reports), stop_reason,
    )
    return summary


def save_batch_summary(summary: dict[str, Any], output_dir: str | Path) -> None:
    """Write batch summary as JSON and markdown.

    Args:
        summary: Batch summary dict.
        output_dir: Directory to write reports to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_id = summary["batch_id"]

    # JSON
    json_path = output_dir / f"batch-{batch_id}-summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved batch summary JSON: %s", json_path)

    # Markdown
    md_path = output_dir / f"batch-{batch_id}-summary.md"
    md_content = format_markdown_summary(summary)
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info("Saved batch summary markdown: %s", md_path)


def format_markdown_summary(summary: dict[str, Any]) -> str:
    """Format a batch summary as human-readable markdown.

    Args:
        summary: Batch summary dict.

    Returns:
        Markdown string.
    """
    lines = [
        f"# Batch Summary: {summary['batch_id']}",
        "",
        "## Metadata",
        "",
        f"- **Iterations:** {summary['n_iterations']}",
        f"- **Stop reason:** {summary['stop_reason']}",
        f"- **Start:** {summary.get('start_time', 'N/A')}",
        f"- **End:** {summary.get('end_time', 'N/A')}",
        f"- **Git commit:** `{summary.get('git_commit', 'unknown')}`",
        "",
    ]

    # Starting vs final config comparison
    starting = summary.get("starting_config", {})
    final = summary.get("final_config", {})
    config_changes = _compute_config_diff(starting, final)
    lines.append("## Config: Start vs End")
    lines.append("")
    if config_changes:
        lines.append("| Parameter | Start | End |")
        lines.append("|-----------|-------|-----|")
        for change in config_changes:
            lines.append(f"| `{change['key']}` | {change['old']} | {change['new']} |")
    else:
        lines.append("No config changes (same config start to finish).")
    lines.append("")

    # Full starting config for reference
    lines.append("**Starting config:**")
    lines.append("")
    for key, val in sorted(starting.items()):
        lines.append(f"- `{key}`: {val}")
    lines.append("")

    # Config trajectory (iteration-by-iteration changes)
    config_traj = summary.get("config_trajectory", [])
    has_changes = any(entry.get("changes") for entry in config_traj)
    if has_changes:
        lines.append("## Config Trajectory (Per-Iteration)")
        lines.append("")
        lines.append("| Iteration | Key | Old | New |")
        lines.append("|-----------|-----|-----|-----|")
        for entry in config_traj:
            for change in entry.get("changes", []):
                lines.append(
                    f"| {entry['iteration']} | `{change['key']}` | "
                    f"{change['old']} | {change['new']} |"
                )
        lines.append("")

    # Metric trajectory
    lines.append("## Metric Trajectory")
    lines.append("")
    metric_traj = summary.get("metric_trajectory", {})
    if metric_traj:
        n_iter = summary["n_iterations"]
        lines.append(f"| Metric | Trend | " + " | ".join(f"Iter {i}" for i in range(n_iter)) + " |")
        lines.append("|--------|-------| " + " | ".join("---" for _ in range(n_iter)) + " |")
        for metric_name, data in sorted(metric_traj.items()):
            values_str = " | ".join(
                f"{v:.4f}" if v is not None else "N/A"
                for v in data["values"]
            )
            lines.append(f"| {metric_name} | {data['trend']} | {values_str} |")
        lines.append("")

    # Per-family breakdown (from final iteration)
    per_family = summary.get("per_family", {})
    if per_family:
        lines.append("## Per-Family Breakdown (Final Iteration)")
        lines.append("")
        lines.append("| Family | N | MR-STFT (mean) | MFCC (mean) | Acceptance | vs Baseline |")
        lines.append("|--------|---|----------------|-------------|------------|-------------|")
        for family_name, fdata in sorted(per_family.items()):
            n = fdata.get("n_samples", 0)
            mrstft = fdata.get("metrics", {}).get("mrstft", {}).get("mean", "N/A")
            mfcc = fdata.get("metrics", {}).get("mfcc", {}).get("mean", "N/A")
            acc = fdata.get("acceptance_rate", "N/A")
            bl_comp = fdata.get("baseline_comparison", {})
            bl_str = ""
            for metric_name, bl in bl_comp.items():
                in_band = bl.get("in_band", "N/A")
                bl_median = bl.get("baseline_median", "N/A")
                bl_str += f"{metric_name}: {'in-band' if in_band else 'OUT'} (bl median={bl_median:.3f}) " if isinstance(bl_median, (int, float)) else f"{metric_name}: {in_band} "
            mrstft_str = f"{mrstft:.3f}" if isinstance(mrstft, (int, float)) else str(mrstft)
            mfcc_str = f"{mfcc:.1f}" if isinstance(mfcc, (int, float)) else str(mfcc)
            acc_str = f"{acc:.0%}" if isinstance(acc, (int, float)) else str(acc)
            lines.append(f"| {family_name} | {n} | {mrstft_str} | {mfcc_str} | {acc_str} | {bl_str.strip()} |")
        lines.append("")

    # Best/worst samples with per-sample metrics
    lines.append("## Best Samples")
    lines.append("")
    for s in summary.get("best_samples", []):
        lines.append(f"- **{s['name']}** (score: {s.get('score', 'N/A'):.4f}, iter: {s.get('iteration', 'N/A')})")
    lines.append("")

    lines.append("## Worst Samples")
    lines.append("")
    for s in summary.get("worst_samples", []):
        lines.append(f"- **{s['name']}** (score: {s.get('score', 'N/A'):.4f}, iter: {s.get('iteration', 'N/A')})")
    lines.append("")

    # Claude diagnoses
    diagnoses = summary.get("claude_diagnoses", [])
    if diagnoses:
        lines.append("## Claude Diagnoses")
        lines.append("")
        for d in diagnoses:
            lines.append(f"### Iteration {d['iteration']}")
            lines.append("")
            lines.append(d["reasoning"])
            lines.append("")

    # Listening notes (pre-filled or template)
    lines.append("## Listening Notes")
    lines.append("")
    if summary.get("listening_notes"):
        lines.append(summary["listening_notes"])
    else:
        lines.append("_To be filled in after auditioning the listening pack._")
        lines.append("")
        lines.append("### Machine Gun Test")
        lines.append("")
        lines.append("Compare `machinegun_source.wav` (same hit repeated) vs `machinegun_variations.wav` (ML variations):")
        lines.append("")
        lines.append("| Sample | Source MG | Variations MG | Notes |")
        lines.append("|--------|----------|---------------|-------|")
        for s in summary.get("best_samples", []):
            lines.append(f"| {s['name']} | | | |")
        for s in summary.get("worst_samples", []):
            if s['name'] not in {bs['name'] for bs in summary.get("best_samples", [])}:
                lines.append(f"| {s['name']} | | | |")
        lines.append("")
        lines.append("### Overall Impressions")
        lines.append("")
        lines.append("- Identity preservation: ")
        lines.append("- Variation quality: ")
        lines.append("- Problem families: ")
        lines.append("- Recommended next step: ")
    lines.append("")

    return "\n".join(lines)


def assemble_listening_pack(
    batch_id: str,
    iteration_reports: list[dict[str, Any]],
    output_dir: str | Path,
    n_best: int = 5,
    n_worst: int = 5,
) -> Path:
    """Copy best/worst audio files into a listening pack directory.

    Args:
        batch_id: Batch identifier.
        iteration_reports: List of iteration report dicts.
        output_dir: Base output directory.
        n_best: Number of best samples to include.
        n_worst: Number of worst samples to include.

    Returns:
        Path to the listening pack directory.
    """
    output_dir = Path(output_dir)
    pack_dir = output_dir / f"listening-pack" / f"batch-{batch_id}"
    # Clean up stale files from previous runs
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    # Collect all scored samples with their audio paths
    all_samples: list[tuple[str, float, dict]] = []
    for report in iteration_reports:
        audio_paths = report.get("audio_paths", {})
        for sample in report.get("best_samples", []) + report.get("worst_samples", []):
            name = sample["name"]
            if name in audio_paths:
                all_samples.append((name, sample["score"], audio_paths[name]))

    # Deduplicate
    seen: set[str] = set()
    unique: list[tuple[str, float, dict]] = []
    for name, score, paths in all_samples:
        if name not in seen:
            seen.add(name)
            unique.append((name, score, paths))

    unique.sort(key=lambda x: x[1])
    selected = unique[:n_best] + unique[-n_worst:]

    # Copy files
    for name, score, paths_info in selected:
        sample_dir = pack_dir / name
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Copy source
        source_path = Path(paths_info.get("source", ""))
        if source_path.exists():
            shutil.copy2(source_path, sample_dir / "source.wav")

        # Copy variations
        for i, var_path in enumerate(paths_info.get("variations", [])):
            var_path = Path(var_path)
            if var_path.exists():
                shutil.copy2(var_path, sample_dir / f"var_{i + 1:02d}.wav")

        # Render machine-gun comparison from variation files (stereo-aware)
        var_audios_raw = []
        for var_path in paths_info.get("variations", []):
            var_path = Path(var_path)
            if var_path.exists():
                try:
                    from src.utils.audio import load_wav
                    var_audios_raw.append(load_wav(var_path))  # [channels, samples]
                except Exception as e:
                    logger.warning("Failed to load %s for machine-gun render: %s", var_path, e)

        if len(var_audios_raw) >= 2:
            n_ch = var_audios_raw[0].shape[0]
            mg_channels = []
            for ch in range(n_ch):
                hits = [a[ch] for a in var_audios_raw]
                mg_channels.append(render_machine_gun(hits))
            mg_out = np.stack(mg_channels, axis=0)
            save_wav(mg_out, sample_dir / "machinegun_variations.wav")

        # Render source-only machine gun (same hit repeated) for A/B comparison
        if source_path.exists():
            try:
                from src.utils.audio import load_wav
                source_audio = load_wav(source_path)
                mg_src_channels = []
                for ch in range(source_audio.shape[0]):
                    mg_src_channels.append(render_machine_gun([source_audio[ch]]))
                mg_src_out = np.stack(mg_src_channels, axis=0)
                save_wav(mg_src_out, sample_dir / "machinegun_source.wav")
            except Exception as e:
                logger.warning("Failed to render source machine gun for %s: %s", name, e)

    logger.info("Assembled listening pack: %s (%d samples)", pack_dir, len(selected))
    return pack_dir
