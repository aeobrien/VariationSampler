"""Shared instrument-to-family mapping for evaluation and baseline computation."""

import re

# Map specific instrument names to canonical families.
# Used by compute_baselines.py (offline) and the evaluation pipeline (runtime).
INSTRUMENT_FAMILIES: dict[str, str] = {
    "Snare": "Snare",
    "SnareRim": "Snare",
    "Kick": "Kick",
    "HiHatClosed": "HiHat",
    "HiHatOpen": "HiHat",
    "PedalHat": "HiHat",
    "Rimshot": "Rimshot",
    "CrossStick": "CrossStick",
    "Tom": "Tom",
    "TomHi": "Tom",
    "TomLo": "Tom",
    "TomFloor": "Tom",
    "Crash": "Cymbal",
    "Ride": "Cymbal",
}

# Pattern: Track{N}_{Instrument}_v{velocity}
_DIR_PATTERN = re.compile(r"^Track(\d+)_(.+)_v(\d+)$")


def infer_family(group_key: str) -> str | None:
    """Extract instrument from a group key and map to its family.

    Args:
        group_key: Directory name like "Track1_Snare_v127".

    Returns:
        Family name (e.g. "Snare", "HiHat"), or None if unrecognised.
    """
    match = _DIR_PATTERN.match(group_key)
    if not match:
        return None
    instrument = match.group(2)
    return INSTRUMENT_FAMILIES.get(instrument)
