"""Regenerate batch summary and listening pack from existing iteration reports.

Usage:
    python3 scripts/regenerate_summary.py --batch-dir reports/phase4-001 \
        --outputs-dir outputs/phase4-001
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.automation.batch_summary import (
    generate_batch_summary,
    save_batch_summary,
    assemble_listening_pack,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate batch summary from iteration reports.")
    parser.add_argument("--batch-dir", required=True, help="Directory containing iteration-*.json files")
    parser.add_argument("--outputs-dir", required=True, help="Directory containing iteration output audio")
    parser.add_argument("--batch-id", default=None, help="Override batch ID (default: read from first report)")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    outputs_dir = Path(args.outputs_dir)

    # Load iteration reports in order
    report_files = sorted(batch_dir.glob("iteration-*.json"))
    if not report_files:
        logger.error("No iteration-*.json files found in %s", batch_dir)
        sys.exit(1)

    reports = []
    for rf in report_files:
        with open(rf) as f:
            reports.append(json.load(f))
    logger.info("Loaded %d iteration reports from %s", len(reports), batch_dir)

    # Read batch ID from existing summary if available
    batch_id = args.batch_id
    if batch_id is None:
        existing_summaries = list(batch_dir.glob("batch-*-summary.json"))
        if existing_summaries:
            with open(existing_summaries[0]) as f:
                batch_id = json.load(f).get("batch_id", "unknown")
        else:
            batch_id = batch_dir.name

    # Read stop reason from existing summary
    stop_reason = "unknown"
    existing_summaries = list(batch_dir.glob("batch-*-summary.json"))
    if existing_summaries:
        with open(existing_summaries[0]) as f:
            stop_reason = json.load(f).get("stop_reason", "unknown")

    # Regenerate summary
    summary = generate_batch_summary(
        batch_id=batch_id,
        iteration_reports=reports,
        stop_reason=stop_reason,
    )
    save_batch_summary(summary, batch_dir)

    # Regenerate listening pack
    try:
        pack_dir = assemble_listening_pack(
            batch_id=batch_id,
            iteration_reports=reports,
            output_dir=outputs_dir,
        )
        logger.info("Listening pack written to: %s", pack_dir)
    except Exception as e:
        logger.error("Failed to assemble listening pack: %s", e)

    logger.info("Done. Summary written to %s", batch_dir)


if __name__ == "__main__":
    main()
