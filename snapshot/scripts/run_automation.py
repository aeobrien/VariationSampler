"""CLI entry point for the automation runner.

Usage:
    python3 scripts/run_automation.py --batch-config configs/batch-template.json
    python3 scripts/run_automation.py --batch-config configs/batch-template.json --dry-run
    python3 scripts/run_automation.py --batch-config configs/batch-template.json --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.automation.runner import AutomationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the VariationSampler automation loop",
    )
    parser.add_argument(
        "--batch-config",
        type=Path,
        required=True,
        help="Path to batch config JSON file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 iteration without Claude API call",
    )
    args = parser.parse_args()

    if not args.batch_config.exists():
        logger.error("Batch config not found: %s", args.batch_config)
        sys.exit(1)

    # If dry-run requested, patch the config
    if args.dry_run:
        with open(args.batch_config) as f:
            config = json.load(f)
        config["dry_run"] = True
        config["max_iterations"] = 1
        # Write to temp location
        dry_run_path = args.batch_config.parent / f"{args.batch_config.stem}_dryrun.json"
        with open(dry_run_path, "w") as f:
            json.dump(config, f, indent=2)
        args.batch_config = dry_run_path
        logger.info("Dry run mode: 1 iteration, no Claude API")

    # Auto-detect device if not specified
    device = args.device
    if device is None:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    logger.info("Starting automation runner (device=%s)", device)

    runner = AutomationRunner(args.batch_config, device=device)
    summary = runner.run()

    logger.info(
        "Batch '%s' finished: %d iterations, stop reason: %s",
        summary.get("batch_id", "?"),
        summary.get("n_iterations", 0),
        summary.get("stop_reason", "?"),
    )


if __name__ == "__main__":
    main()
