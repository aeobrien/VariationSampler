# Training Data: Capture Overview & Processing Requirements

## How the Data Was Captured

Samples were batch-captured from **Addictive Drums 2** (AD2) running in Reaper using a Lua script. Multiple AD2 instances were loaded on separate tracks, each with a different drum kit. The script programmed MIDI hits for each instrument at multiple velocities with 10 repeated hits per velocity layer, then rendered each hit as a separate WAV file via Reaper's region rendering.

AD2 internally uses round-robin sample playback — when the same note is triggered repeatedly at the same velocity, it cycles through its internal round-robin pool (typically 3-6 recordings). By capturing 10 hits per velocity, we oversample to ensure we capture the full round-robin cycle, then detect and remove duplicates in preprocessing.

## File Naming Convention

The Reaper script named each render region (and therefore each output file) as:

```
{TrackName}_{Instrument}_v{velocity}_rr{hitNumber}.wav
```

**However, tracks were not renamed from Reaper defaults**, so files begin with `Track 1`, `Track 2`, etc. rather than kit names.

### Examples of actual filenames:
```
Track 1_Kick_v025_rr01.wav
Track 1_Kick_v025_rr02.wav
...
Track 1_Kick_v025_rr10.wav
Track 1_Kick_v051_rr01.wav
...
Track 1_Snare_v127_rr10.wav
Track 2_Kick_v025_rr01.wav
...
```

### How to parse the filename:

| Field | Meaning | How to extract |
|---|---|---|
| TrackName | Identifies the drum kit (AD2 instance). Currently `Track 1`, `Track 2`, etc. | Everything before the first `_` that matches an instrument name |
| Instrument | Drum type | One of: `Kick`, `Snare`, `CrossStick`, `Rimshot`, `HiHatClosed` |
| Velocity | MIDI velocity (0-127) | Three-digit zero-padded number after `_v` |
| HitNumber | Round-robin capture index (1-10) | Two-digit zero-padded number after `_rr` |

**Note on track names**: Because tracks weren't named, `Track 1` through `Track N` serve as opaque kit identifiers. Each track number maps to a unique AD2 kit. The actual kit identity doesn't matter for training — what matters is that all files sharing the same `TrackName + Instrument + Velocity` belong to the **same round-robin set** (same drum, same velocity, different hits).

### Group key for splitting/training:

```
group_key = (TrackName, Instrument, Velocity)
```

For example, all files matching `Track 1_Snare_v076_rr*.wav` form one round-robin set.

The `rr` number is just a capture index — it does NOT indicate which internal AD2 round-robin sample was triggered. That's why duplicate detection is needed (see below).

## Instruments and MIDI Notes

| Instrument | MIDI Note | AD2 Default |
|---|---|---|
| Kick | 36 | C1 |
| CrossStick | 37 | C#1 (side stick) |
| Snare | 38 | D1 |
| Rimshot | 40 | E1 |
| HiHatClosed | 42 | F#1 |

## Velocity Layers

| Layer | MIDI Velocity |
|---|---|
| 1 (very soft) | 25 |
| 2 (soft) | 51 |
| 3 (medium) | 76 |
| 4 (hard) | 102 |
| 5 (full) | 127 |

## Processing Steps Required

### 1. Duplicate Detection and Removal

AD2 has a finite round-robin pool per instrument/velocity (typically 3-6 unique samples). With 10 captures per set, some will be duplicates where AD2 cycled back to the same internal sample. These need to be identified and removed.

**Approach**: Within each round-robin set (same group_key), compute pairwise waveform similarity. Samples that are near-identical (below a tight threshold) are duplicates. Keep one copy of each unique sample.

**Method**: Short-window cross-correlation on the attack region (first 20-50ms) is the most reliable for this. Duplicates from the same internal AD2 sample will have correlation > 0.999. Natural round-robin variations will be high (0.95-0.99) but not near-perfect.

**Expected outcome**: Each set of 10 captures should reduce to 3-6 unique hits. Log the duplicate count per set — if a set reduces to only 1-2 unique hits, that instrument/velocity may not have meaningful round-robin variation in AD2 and should be flagged.

### 2. Leading Silence Trimming

Each file has a small amount of silence before the transient attack. This comes from two sources:
- The Reaper capture region starts 50ms before the MIDI trigger (intentional pre-hit pad)
- AD2's own latency between receiving MIDI and producing audio

**Trimming approach** (per the technical brief):
- Use spectral flux onset detection to find the transient
- Backtrack slightly to preserve any pre-transient noise (this is part of the realism signature — don't hard-cut right at the transient)
- Trim conservatively: keep a small margin before the detected onset

**Important**: Do NOT trim aggressively. The technical brief explicitly states "never clip pre-transient content." A few ms of room noise before the hit is fine and natural.

### 3. Transient Alignment

After trimming, all samples within a round-robin set should be aligned to a consistent onset point. This ensures the model learns timbral variation, not timing variation.

Per the technical brief's preprocessing pipeline:
1. Coarse onset via spectral flux onset strength (librosa)
2. Backtrack to pre-onset reference
3. Fine alignment via short-window cross-correlation on attack band
4. Align to onset point, not absolute peak

### 4. Pad/Truncate to Fixed Length

Before setting the fixed pad/truncate length, analyse the captured samples to find where energy actually drops to negligible levels. For each instrument type, compute the RMS energy envelope and find the time at which it drops below, say, -60dB relative to peak. Report the distribution (median, 95th percentile) per instrument. That tells us what the pad length should actually be. The technical brief assumed 1.0s but some samples clearly extend beyond that.

### 5. Loudness Normalisation

Normalise within ±1 dB tolerance within each round-robin set. This reduces level as a confound without destroying natural micro-dynamics. Do not over-normalise.

### 6. DAC Encoding and Caching

Encode all processed audio through DAC and cache codegrams as tensors. This is the final preprocessing step before training.

## Directory Structure Suggestion

After processing, organise as:

```
data/
├── raw/                          # Original captured WAVs, untouched
│   └── pass-01/
│       ├── Track 1_Kick_v025_rr01.wav
│       ├── Track 1_Kick_v025_rr02.wav
│       └── ...
├── deduplicated/                 # After duplicate removal
│   └── pass-01/
│       └── ...
├── processed/                    # Trimmed, aligned, normalised, padded
│   └── {group_key}/             
│       ├── hit_01.wav
│       ├── hit_02.wav
│       └── ...
├── codegrams/                    # DAC-encoded tensors
│   └── {group_key}.pt
├── baselines/                    # Ground-truth metric distributions
│   └── snare_distances.json
│   └── kick_distances.json
│   └── ...
├── manifests/
│   └── manifest.csv              # Full metadata for every sample
└── splits/
    ├── train.csv
    ├── dev.csv
    └── test.csv
```

## Manifest CSV Schema

```csv
filepath,group_key,track_name,instrument,velocity,rr_index,is_duplicate,set_size,pass
Track 1_Kick_v025_rr01.wav,"Track 1_Kick_v025",Track 1,Kick,25,1,false,4,pass-01
```

Where `set_size` is the number of unique (non-duplicate) hits in the round-robin set, populated after deduplication.

## Data Scale Summary

Per capture pass with 20 AD2 instances:

| Metric | Count |
|---|---|
| Kits (tracks) | 20 |
| Instruments per kit | 5 |
| Velocity layers | 5 |
| **Round-robin sets** | **500** |
| Hits per set (before dedup) | 10 |
| Expected unique hits per set | 3-6 |
| **Total unique samples (estimated)** | **1,500 - 3,000** |
| Training pairs per set (N unique, N*(N-1) ordered pairs) | ~12-30 |
| **Total training pairs (estimated)** | **6,000 - 15,000** |

This lands in the "Minimum viable (POC)" range from the technical brief (500-1,500 sets, 10k-30k pairs). A second pass with different kits doubles it into the "Recommended" range.