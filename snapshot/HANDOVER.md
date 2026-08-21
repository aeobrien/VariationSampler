# Variation Sampler — Session Handover

**Date:** 2026-03-03
**Purpose:** Complete context for continuing work. Start the next conversation by pointing to this file.

---

## What This Project Does

Variation Sampler generates perceptually convincing round-robin variations from a single one-shot drum sample. Given one recording of a drum hit, it produces 4-8 variations that sound like the same instrument struck again — breaking the "machine gun effect" of identical sample repetition.

The approach: frozen DAC audio codec as tokenizer/decoder + a masked-token micro-inpainting transformer + audio-domain acceptance filtering.

---

## Current State: Phase 4 (Calibration & Tuning)

**Gate A: PASSED.** The POC is validated — ML micro-inpainting produces convincing drum variations.

Phases 1-3 are complete. 261 tests passing. The model trains, generates, and evaluates. The automation loop runs unattended batches on GCP.

**Phase 4 is in progress.** Three autonomous batch runs completed (phase4-001 through phase4-003). We hit a critical problem and diagnosed it. The immediate next step is to retrain and re-test.

---

## The Critical Discovery: Mask Rate Mismatch Is The Mechanism

This is the single most important thing to understand:

### How variation works

The model is trained at **1x mask rates** (conservative: `p_tail=0.08`). At inference, we use **4x mask rates** (`p_tail=0.32`). The model sees mask positions it rarely encountered during training. Its **prediction uncertainty** at those unfamiliar positions is what produces variation — it distributes probability across plausible tokens rather than confidently predicting the original.

### What went wrong

We retrained the model at 4x mask rates (thinking it would make "better predictions"). It did — the model learned to confidently predict the original tokens at 4x mask positions. Result: **zero perceptible variation**. All four strategies (A/B/C/D) in the strategy comparison test sounded identical to the source. The user confirmed this in a blind listening test.

### The fix

Go back to training at 1x mask rates. The epoch 2 checkpoint from the original 1x training was the best. But it was overwritten by the 4x retrain. **We need to retrain with 1x mask rates to get it back.**

This design property is now documented in:
- `CLAUDE.md` — Critical Design Decision #6
- `CLAUDE.md` — Mistakes to Avoid (last entry)
- `checkpoints/README.md` — full explanation
- `configs/train-1x.yaml` — header comment

**Never retrain at higher mask rates. Train at 1x, infer at 4x.**

---

## The Other Problem: DAC Decoder Frame Coupling

Phase 4 batch runs revealed that CrossStick and SnareRim samples had persistent attack degradation even with `mask_p_attack=0.0`. Investigation showed the DAC decoder couples adjacent frames — editing tail tokens causes bleed into the attack region.

### Diagnostic results (from `scripts/dac_coupling_diagnostic.py`)

- **Backward bleed** (tail edits degrade attack): reaches ~5 frames. xcorr drops at distance 0, recovers by distance 7.
- **Forward bleed** (attack edits degrade tail): reaches 2-3 frames.
- No SnareRim samples were found in the codegram data (only CrossStick, HiHat, Kick).

### Solution: Buffer zone masking (Strategy B)

A ramp zone after `attack_frames` where mask probability increases gradually:
- Frame `attack_frames + 1`: `p_tail * 0.25 * mult`
- Frame `attack_frames + 2`: `p_tail * 0.50 * mult`
- Frame `attack_frames + 3`: `p_tail * 0.75 * mult`
- Frame `attack_frames + 4+`: full `p_tail * mult`

Per-family ramp widths: CrossStick=5, SnareRim=5, HiHat=3, Kick=3.

This hasn't been tested yet with the correct (1x-trained) checkpoint. That's the immediate next step.

---

## What Needs to Happen Next

### Step 1: Retrain at 1x mask rates

The training config `configs/train-1x.yaml` already exists and is correct. On the cloud server:

```bash
# 1. Start the VM
gcloud compute instances start variation-sampler --zone=europe-west4-c

# 2. Upload code (deploy script nesting bug is now fixed)
./scripts/deploy_cloud.sh --upload

# 3. SSH in
gcloud compute ssh variation-sampler --zone=europe-west4-c --tunnel-through-iap

# 4. On the server: rename the current (4x) checkpoint
cd ~/variation-sampler
mv checkpoints/best.pt checkpoints/v2-4x-trained-best.pt

# 5. Train with 1x config (should converge in 2-3 epochs, ~5 minutes)
python3 scripts/train.py --config configs/train-1x.yaml --num-workers 4

# 6. Rename the new checkpoint
cp checkpoints/best.pt checkpoints/v1-1x-trained-best.pt
```

### Step 2: Run strategy comparison with 1x checkpoint

```bash
# Still on the server:
python3 scripts/strategy_comparison.py \
    --checkpoint checkpoints/v1-1x-trained-best.pt \
    --config configs/default.yaml \
    --output-dir outputs/strategy_comparison
```

### Step 3: Download results and run blind listening test

```bash
# From local machine:
gcloud compute scp --recurse --tunnel-through-iap \
    variation-sampler:~/variation-sampler/outputs/strategy_comparison/ \
    ./outputs/strategy_comparison/ \
    --zone=europe-west4-c

# Serve and listen
cd outputs/strategy_comparison && python3 -m http.server 8080
# Open http://localhost:8080/blind_listening_test.html
```

The blind test randomizes strategy labels (A/B/C/D) so you can rank without bias. Save results, then reveal labels.

### Step 4: Stop the VM

```bash
gcloud compute instances stop variation-sampler --zone=europe-west4-c
```

---

## The Four Strategies Being Compared

| ID | Strategy | How It Works |
|----|----------|-------------|
| A | Baseline | Standard masking + default composite score selection |
| B | Buffer zone | Ramp masking (gradual mask probability after attack) + default scoring |
| C | Attack-scored | Standard masking + selection by attack quality score |
| D | Strict acceptance | Standard masking + tighter thresholds (xcorr>0.95, smear>0.95, hf<3dB) |

Strategy B is the hypothesis: it should preserve attack quality (via the buffer zone) while still producing good variation in the tail (via the 4x mask rates on the 1x-trained model).

---

## Key Files

### Configs
| File | Purpose |
|------|---------|
| `configs/train-1x.yaml` | **Training config** — 1x mask rates, always use this for training |
| `configs/default.yaml` | **Inference config** — 4x mask rates, use for generation |
| `configs/batch-phase4.json` | Automation batch config for phase 4 runs |

### Scripts (created this session)
| File | Purpose |
|------|---------|
| `scripts/dac_coupling_diagnostic.py` | Measures bidirectional DAC decoder frame coupling |
| `scripts/strategy_comparison.py` | Generates audio for 4 strategies, outputs manifest.json |
| `tools/blind_listening_test.html` | Self-contained blind listening test page |

### Tests
| File | Tests |
|------|-------|
| `tests/test_strategy_comparison.py` | 20 tests covering buffer zone mask, attack scoring, strict acceptance, acceptance rate tracking, family ramp widths |

### Documentation
| File | Purpose |
|------|---------|
| `checkpoints/README.md` | Explains checkpoint naming and the 1x/4x mechanism |
| `CLAUDE.md` | Updated with design decision #6 and mask rate mistake-to-avoid |

---

## Phase 4 Batch Run History

### phase4-001
- Config: epoch 2 checkpoint (1x-trained), 4x mask rates, default acceptance
- Result: Good variation for Kick/Tom. CrossStick and HiHat had attack degradation.
- Action: Reduced `mask_p_attack` and tightened attack thresholds.

### phase4-002
- Config: Same checkpoint, reduced `mask_p_attack`
- Result: CrossStick attack still degraded even with `mask_p_attack=0.02`
- Action: Set `mask_p_attack=0.0` for next run.

### phase4-003
- Config: `mask_p_attack=0.0`
- Result: CrossStick/SnareRim **still** degraded. Zero attack masking didn't help because the DAC decoder couples adjacent frames — tail edits bleed into the attack region.
- Root cause confirmed: DAC decoder coupling, not masking.

### 4x retrain attempt
- Retrained model at 4x mask rates (100 epochs, best at epoch 27)
- Result: **Zero variation.** Model learned to perfectly predict tokens at 4x positions.
- This was the wrong approach. Now documented as a mistake-to-avoid.

---

## Infrastructure

- **Cloud**: GCP `europe-west4-c`, VM `variation-sampler`, L4 GPU
- **Image**: `pytorch-2-7-cu128-ubuntu-2204-nvidia-570`
- **SSH**: Always needs `--tunnel-through-iap`
- **VM uses**: `python3`/`pip3` (not `python`/`pip`)
- **Deploy script** (`scripts/deploy_cloud.sh`): nesting bug now fixed (was creating `src/src/` on upload)
- **Codegrams**: ~14K files, uploaded via tar archive (not individual SCP)

---

## Known Bugs Fixed This Session

1. **`sample_tokens` TypeError** in strategy_comparison.py — was passing config dict instead of temperature/top_p floats.
2. **Blind test HTML single-hit player 404** — was trying string manipulation on the machinegun path. Now uses `source_single` field from manifest.
3. **Deploy script nesting** — `scp --recurse` was copying `src/` INTO `src/`, creating `src/src/`. Fixed by removing trailing slashes.
4. **Strategy comparison now saves source single hit** — `source_single.wav` alongside `source_machinegun.wav`.

---

## Test Suite

261 tests total, all passing. Run with:

```bash
# Fast tests (no GPU, no codec)
pytest tests/ -m "not slow and not gpu"

# Strategy comparison tests only
pytest tests/test_strategy_comparison.py -v
```

---

## What Success Looks Like

After the 1x retrain + strategy comparison:
- Strategy B (buffer zone) should produce **perceptible variation** (unlike the 4x-trained model)
- Strategy B should have **cleaner attacks** than Strategy A (the baseline that caused phase4 attack degradation)
- CrossStick and SnareRim are the acid test — these are the families that had the worst attack degradation

If Strategy B wins the blind test, integrate buffer zone masking into the mainline code. If inconclusive, refine ramp widths and re-test.
