# Variation Sampler — Development Roadmap

---

## Phase Overview

```
Phase 0: Research & Open-Source Evaluation ─────────────────── [~3–5 days]
    │
Phase 0.5: Procedural Baseline ─────────── GATE 0 ─────────── [~1 weekend]
    │
Phase 1: Environment & Data Pipeline ──────────────────────── [~5–10 days]
    │
Phase 2: Baseline Model & First Outputs ───── GATE A ──────── [~3–7 days]
    │
Phase 3: Metrics, Baselines & Automation Infrastructure ───── [~5–7 days]
    │
Phase 4: Calibration & Tuning ─────────────── GATE B ──────── [~2–4 weeks]
    │                                                     (includes autonomous batches)
Phase 5: Cross-Library Generalisation ─────── GATE C ──────── [~1–2 weeks]
    │
Phase 6: Polish & Delivery ────────────────────────────────── [~3–5 days]
```

**Total estimated timeline:** 8–12 weeks from Phase 0 start to V1 complete (or clear no-go). Could be as short as 1 weekend if Gate 0 proves the procedural baseline is sufficient.

---

## Phase 0: Research & Open-Source Evaluation

**Goal:** Determine whether to build on an existing open-source codebase or start from scratch. Evaluate TRIA, VampNet, and any other relevant systems. Identify any knowledge gaps not covered by the two research reports.

**Why this phase exists:** The architecture is settled (masked-token micro-inpainting on DAC codegrams), but there may be existing implementations that provide significant scaffolding — DAC integration, transformer token prediction, masked sampling mechanics. Using proven code for these components reduces risk and accelerates Phase 2. Equally, if existing codebases are too tightly coupled to their own use cases, adapting them could cost more time than building from scratch. This phase makes that determination.

### 0.1 Evaluate TRIA (Highest Priority)

TRIA (2025) is a masked-token transformer that generates drum audio using DAC codegrams with SoundStorm-style coarse-to-fine unmasking. It is the closest published system to what we need.

**Evaluate:**
- Is the codebase open-source and accessible? What licence?
- DAC integration: how does it handle encoding, decoding, codegram shapes? Can we reuse this directly?
- Transformer architecture: embedding strategy (sum codebook embeddings per frame), attention, projection heads. How closely does this match our spec?
- Sampling/unmasking mechanics: mask scheduling, temperature control, iterative refinement. Reusable?
- What would we need to change? (Remove rhythm conditioning, add input-hit conditioning, restrict to late codebooks, add acceptance filtering)
- Estimated effort to adapt vs build from scratch?

### 0.2 Evaluate VampNet

VampNet (2023) is masked acoustic token modeling for music variation using a bidirectional transformer. Its prompting and sampling design is directly relevant.

**Evaluate:**
- Codebase availability and licence
- Codec integration (VampNet uses EnCodec, not DAC — how hard is swapping?)
- Masking and sampling mechanics: multiple passes, temperature control, variation magnitude dial
- What's reusable vs what's tightly coupled to its music generation use case?

### 0.3 Survey Other Candidates

Brief survey of any other open-source audio token generation systems that might provide useful scaffolding. Check:
- AudioCraft (Meta) — EnCodec + MusicGen codebase. Likely too heavyweight but worth checking for utility code.
- Any DAC-based generation repos on GitHub
- The ICLR 2026 discrete diffusion token inpainting work (if code is released)

### 0.4 Knowledge Gap Assessment

Review the two research reports against what we need to actually build. Identify any gaps:
- Are there implementation details for DAC integration (encoding, decoding, handling stereo, padding) that we need to research further?
- Is the masked token transformer architecture specified in enough detail to implement, or do we need to study TRIA/VampNet architecture code more closely?
- Are there practical PyTorch patterns for this kind of model (codebook embedding, masked prediction) that we should understand before building?

### 0.5 Decision: Build vs Adapt

Write a brief decision document: `docs/phase-0-decision.md`
- Recommendation: build from scratch, or adapt [specific codebase]
- If adapting: which components to reuse, which to replace, estimated effort
- If building: which reference implementations to study for specific components
- Any additional research needed before Phase 1

### Phase 0 Deliverables
- [ ] TRIA evaluation complete
- [ ] VampNet evaluation complete
- [ ] Other candidates surveyed
- [ ] Knowledge gaps identified and addressed (or flagged for Phase 1)
- [ ] Decision document written
- [ ] Session log updated

---

## Phase 0.5: Procedural Baseline

**Goal:** Build a simple procedural variation generator, evaluate it critically, and document its failure modes. This is the quality floor that every ML output will be compared against. If it's already good enough, the ML approach is unnecessary.

**Why this phase exists:** The confidence assessment rates "consistently beat hand-designed humanisation/randomisation" as medium-low. If a weekend of procedural work already solves the problem, we save weeks. If it doesn't, we get a concrete, articulable description of what's missing — which sharpens every subsequent gate evaluation from "does it sound good?" to "does it sound better than the cheap version, and in what specific ways?"

### 0.5.1 Build Procedural Variation Generator

Apply per-variation randomised:
- Gain: ±0.3 dB
- Pitch: ±3 cents
- Transient shaping: subtle attack time jitter
- Short convolution with micro-impulses for subtle tonal colouring
- Sample-level micro-timing offset: ±5 samples

This should be a standalone Python script that takes a WAV input and produces N variation WAVs.

### 0.5.2 Evaluate

- Render machine-gun tests (16th notes at 120 BPM) across snares, kicks, hats
- Listen critically (project owner)
- A/B against real round-robin sets from commercial libraries
- Document: what does it get right? Where does it fall apart?

### 0.5.3 — GATE 0: Procedural Baseline Evaluation

**Gate criteria:** Is the procedural baseline sufficient for the project owner's production use?

**Possible outcomes:**
- **Project complete:** The procedural baseline is good enough. Ship it. No ML needed.
- **Proceed with ML:** The procedural baseline fails in specific ways (e.g., "timbral variation is always in the wrong dimension — it sounds like EQ, not like a different strike"). Document failure modes. These become the ML model's explicit targets and the benchmark to beat.

**This gate requires explicit project owner decision.**

### Phase 0.5 Deliverables
- [ ] Procedural variation generator working
- [ ] Machine-gun test audio rendered for snares/kicks/hats
- [ ] Listening evaluation complete
- [ ] Failure modes documented (or project declared complete)
- [ ] Gate 0 decision recorded
- [ ] Session log updated

---

## Phase 1: Environment & Data Pipeline

**Goal:** Set up the full development environment, build the data preprocessing pipeline, and produce cached codegrams ready for training. This phase ends with a working pipeline from raw WAV files to cached DAC codegrams with computed baseline metrics.

### 1.1 Environment Setup

- Python environment (venv or conda) with pinned dependencies
- DAC installed and verified: encode a sample WAV → codegram → decode → verify roundtrip quality by listening
- PyTorch with CUDA verified on target GPU
- Weights & Biases account and project created
- Google Cloud setup: VM configuration, storage buckets, SSH access
- Git repository initialised with project structure from CLAUDE.md
- All reference documents moved to `docs/`

**Verification:** DAC roundtrip test — encode a handful of drum samples, decode, listen. Should be perceptually transparent. If DAC introduces audible artifacts on the source material, flag immediately (this would challenge a core assumption). **Also verify stereo handling:** encode a stereo drum sample, decode, check that stereo image is preserved (pan position, width, transient coherence between channels). If stereo reconstruction is lossy, consider mono processing with stereo restored post-generation.

### 1.2 Data Import & Organisation

- Import script that reads commercial library directory structures
- Metadata extraction: library ID, kit ID, instrument ID, articulation, velocity bin, mic perspective
- Organise into the canonical structure: `data/raw/{library}/{kit}/{instrument}/{articulation}/{velocity}/`
- Manifest file (CSV or JSON) mapping every WAV to its full metadata
- Initial count: how many round-robin sets, what N distribution, which instrument families

**Verification:** Print summary statistics. Confirm we have at least 500 snare round-robin sets for the POC.

### 1.3 Preprocessing Pipeline

- Silence trimming with conservative margins (never clip pre-transient content)
- Transient alignment:
  - Spectral flux onset detection (librosa)
  - Backtrack to pre-onset reference
  - Fine alignment via short-window cross-correlation on attack band
- Pad/truncate to 1.0 second
- Loudness normalisation within ±1 dB
- Write processed audio to `data/processed/`

**Verification:** Spot-check 20–30 aligned pairs by visual waveform comparison and listening. Transients should be sample-accurately aligned. Pre-transient noise should be preserved.

### 1.4 DAC Encoding & Codegram Cache

- Encode all processed audio through DAC
- Save codegrams as PyTorch tensors: `data/codegrams/{group_key}.pt`
- Verify codegram shapes match expected `[Nq, T_max]` = `[9, 86]`
- Build a dataset class that loads cached codegrams by group key

**Verification:** Decode a random sample of cached codegrams back to audio. Listen. Should be identical to the processed WAV (within DAC's reconstruction quality).

### 1.5 Splits & Training Pair Generation

- Implement splitting protocol: split by group key `(library_id, kit_id, instrument_id, articulation_id, mic_perspective_id)`
- No group key in more than one split
- Splits: train, dev (same libraries, different instruments), test (held-out entire libraries)
- Training pair generation: for each round-robin set with N hits, generate N*(N-1) ordered pairs
- Verify no leakage: audit that no group key appears in multiple splits

**Verification:** Print split statistics. Confirm test split contains entire held-out libraries.

### 1.6 Ground-Truth Baseline Metrics

- Compute within-set pairwise distances for every round-robin set in training data:
  - Multi-resolution STFT distance (spectral convergence + log magnitude)
  - MFCC distance
- Aggregate into distributions by instrument family: median, IQR, p5, p95
- Save to `data/baselines/`
- These become the target bands for acceptance filtering and the reference for "is our variation the right magnitude?"

**Verification:** Visualise distributions. Sanity-check: are kick distances different from snare distances? (They should be — different spectral characteristics.)

### Phase 1 Deliverables
- [ ] Environment fully set up and verified (DAC roundtrip clean)
- [ ] Data imported with full metadata manifest
- [ ] Preprocessing pipeline working (trim, align, normalise, pad)
- [ ] Codegrams cached for full dataset
- [ ] Splits generated with no leakage
- [ ] Baseline metric distributions computed and saved
- [ ] All tests passing
- [ ] Session log updated

---

## Phase 2: Baseline Model & First Outputs

**Goal:** Build the micro-inpainting transformer, train it on snare data, and produce the first listenable variations. This phase ends at **Gate A**: do the outputs preserve transient integrity?

### 2.1 Model Architecture

Build the masked-token micro-inpainting transformer:

- Codebook embedding tables: `E_i: [1024] → R^D` for each of 9 codebooks
- Frame representation: sum embeddings across codebooks per time step (TRIA approach)
- Mask embedding: learned embedding for masked positions
- Bidirectional transformer encoder (start small: D=256, 4–6 layers, 4–8 heads)
- Per-codebook projection heads for editable codebooks → logits `[T_max, 1024]`
- Config-driven: all architecture hyperparameters in config file

**Verification:** Forward pass with random input produces correct output shapes. No NaN/Inf.

### 2.2 Training Loop

- Masked token cross-entropy training
- For each batch: sample pairs (A, B) from same round-robin set
- Encode mask: random positions in editable codebooks (6–8), with separate mask probabilities for attack vs tail frames
- Predict masked tokens of z_B conditioned on z_A
- Loss: cross-entropy on masked positions only
- Optimiser: AdamW
- Learning rate schedule: warmup + cosine decay
- W&B logging: loss curves, learning rate, gradient norms

**Verification:** Loss decreases over first 1000 steps. Not diverging, not stuck.

### 2.3 Inference Pipeline

- Load trained checkpoint
- Given input WAV: encode → build stochastic mask → forward pass → sample tokens → replace masked positions → decode
- Acceptance filtering: compute MR-STFT distance to input, reject if outside target band
- Generate K candidates, keep those that pass
- Post-processing chain: DC removal, level matching, tail fade, dither to 16-bit

**Verification:** Produces WAV files that are loadable and have correct format (44.1 kHz, 16-bit, stereo).

### 2.4 First Generation Run

- Train on snare data for initial run (start with ~5000–10000 steps, increase if needed)
- Generate 4–8 variations for 10–20 held-out snare inputs
- Listen to every output
- Log to W&B with audio samples

### 2.5 — GATE A: First Listenable Outputs

**Gate criteria:** Generated variations must preserve transient integrity — no obvious ringing, smeared attacks, or tonal artifacts on snares.

**Gate evaluation:**
1. Generate variations for 20+ held-out snare samples
2. Listen to all outputs (project owner)
3. Specific check: play each variation in isolation — is the transient clean?
4. Specific check: A/B with the input — does it sound like the same drum?
5. Write gate evaluation: `reports/gate-A-evaluation.md`

**Possible outcomes:**
- **Pass:** Transients are clean, output sounds like plausible drum hits. Proceed to Phase 3.
- **Adjust:** Transients are mostly clean but some artifacts. Adjust codebook range, mask probabilities, or acceptance thresholds. Re-evaluate.
- **Fail:** Systematic transient destruction despite late-codebook-only editing and acceptance filtering. Investigate whether DAC reconstruction itself is the bottleneck (re-run roundtrip test). If DAC is clean but variations are not, the model may need architectural changes. If neither works, consider pivot.

**This gate requires explicit project owner approval to pass.**

### Phase 2 Deliverables
- [ ] Model architecture implemented and verified
- [ ] Training loop working with W&B logging
- [ ] Inference pipeline producing WAV files
- [ ] First generation run complete
- [ ] Gate A evaluation document written
- [ ] Project owner has listened and approved (or directed adjustments)
- [ ] Session log updated

---

## Phase 3: Metrics, Evaluation & Automation Infrastructure

**Goal:** Build the full evaluation suite, the machine-gun proxy test, and the autonomous iteration loop. Pre-Gate A, iteration was manual. Now that we have a working model, build the automation that enables overnight/weekend tuning runs.

**Why now, not earlier:** The automation loop is only valuable once there's a working model to iterate on. Building it before Gate A would be premature — we'd be building infrastructure for something that might not produce listenable output. Now that Gate A has confirmed the approach works, longer unattended tuning runs become the bottleneck.

### 3.1 Full Metric Suite

Implement all evaluation metrics:

**Similarity metrics:**
- Multi-resolution STFT distance (3 resolutions: 10ms/2ms, 25ms/5ms, 50ms/10ms)
  - Compute on attack window (first 30–80ms) AND full clip separately
- Multi-resolution complex STFT distance (phase-aware, for layering safety)
- MFCC distance
- Token change rate: `mean(z_out != z_in)` on editable codebooks

**Variation metrics:**
- Inter-variation pairwise distances (same metrics as similarity, computed between generated variations)
- Comparison to ground-truth baseline distributions (median, IQR)

**Anti-gaming metrics:**
- Attack smear score: ratio of transient energy pre/post variation (should be ~1.0)
- Tonal ringing detector: spectral peaks in variation that don't exist in input
- High-frequency energy delta: energy above 12 kHz compared to input (hard-reject if delta exceeds threshold)
- Transient cross-correlation: cross-correlate attack region of input and variation (should be >0.95)

**Verification:** Run metrics on real round-robin pairs. Results should match the baseline distributions computed in Phase 1.

### 3.2 Machine-Gun Proxy Test

- Render 8-hit sequence at 16th notes, 120 BPM
- Extract per-hit features (MFCCs over first 50–100ms + envelope/centroid/flux)
- Compute self-similarity matrix
- Score: mean cosine similarity for i≠j
- Compute for: identical copies (should be ~1.0), real round-robin sets (baseline), generated variations, AND procedural baseline variations (from Gate 0)

**Verification:** Identical copies score near 1.0. Real RR sets score lower. The metric discriminates. Procedural baseline establishes the quality floor.

### 3.3 Iteration Report Generator

Build the JSON report writer that produces the structured report after each training/evaluation iteration:

- Run metadata (ID, git commit, timestamp, config snapshot)
- All metric values (mean, p95, distribution stats)
- Comparison to baselines (in-band? above? below?)
- Comparison to procedural baseline scores
- Trend vs previous iterations
- Best and worst sample paths
- Acceptance rate

**Verification:** Generate a report from the Phase 2 training run. Confirm it's valid JSON, all fields populated, audio paths correct.

### 3.4 Batch Summary Report Generator

Build the batch summary that aggregates across all iterations in an autonomous batch:

- Config trajectory (what changed each iteration)
- Metric trajectory (trends across iterations)
- Best/worst outputs across the batch
- All Claude diagnoses and reasoning
- Listening pack assembly: organise audio files for easy human review
- Machine-gun test audio: render rapid-sequence WAVs for each test sample

**Verification:** Generate a mock batch summary from Phase 2 data.

### 3.5 Claude API Integration

Build the module that sends iteration reports to Claude and receives config updates:

- Format the prompt: system instructions (what Claude's role is, what it may change, maximum step sizes), iteration report JSON, optional listening notes from previous batch, trend data
- Parse response: extract config JSON, validate all values are within allowed ranges and step sizes
- Require `reasoning` field in response (reduces erratic hyperparameter leaps)
- Guardrails: reject any response that tries to change non-config values, flag unexpected keys, enforce per-parameter step size limits
- Logging: log every prompt and response for audit

**Verification:** Test with a mock report. Confirm Claude returns valid config JSON with reasoning. Confirm guardrails reject invalid responses and over-sized steps.

### 3.6 Automation Runner

Build the top-level runner script:

- Reads batch config (starting hyperparameters, stopping conditions, guardrail thresholds, per-parameter step size limits)
- Runs the iteration loop: train → evaluate → report → Claude → validate → apply config → repeat
- Implements stopping conditions (iteration cap, stagnation, regression, error)
- Implements auto-rollback on metric regression
- Writes batch summary on stop
- Assembles listening pack

**Verification:** Run a mini-batch (2–3 iterations) end-to-end on a small subset of data. Confirm the loop runs, stops correctly, produces reports and listening pack. **If this takes more than 2 days to build, stop and continue with manual iteration — revisit automation later.**

### 3.7 OOD Evaluation Pack

- Assemble ~50–200 one-shot found sounds (metal hits, paper, plastic, foley) with no round-robin ground truth
- These are for evaluating generalisation — not used in training
- Store in `data/ood-eval/`
- Note: OOD evaluation is purely subjective (listening-only gate). Metrics provide sanity checks but cannot validate plausible re-strike quality without ground truth.

### Phase 3 Deliverables
- [ ] Full metric suite implemented and verified against real RR data
- [ ] Machine-gun proxy test working and discriminating
- [ ] Iteration report generator producing valid structured reports
- [ ] Batch summary report generator working
- [ ] Claude API integration working with guardrails and step size limits
- [ ] Automation runner tested end-to-end (mini-batch) — or deferred if >2 days
- [ ] OOD evaluation pack assembled
- [ ] All tests passing
- [ ] Session log updated

---

## Phase 4: Calibration & Tuning

**Goal:** Use the autonomous iteration loop to find hyperparameter settings that produce variations matching real round-robin magnitude while preserving source identity. This phase ends at **Gate B**: can the model simultaneously satisfy both axes (similar enough AND different enough)?

This is the longest phase. It alternates between autonomous batches and supervised check-in sessions.

### 4.0 Pre-Calibration Setup

Before the first autonomous batch:
- **Varied round-robin set generation mode:** Generate sets with intentionally different mask rates per sample (e.g., 2 at 3x, 2 at 4x, 2 at 6x) to produce more natural-sounding variation within a round-robin group.
- **Retraining at 4x mask rates with early stopping:** Retrain with patience 5 epochs on dev loss to confirm epoch 2 best.pt is still optimal under the updated mask rate defaults.

### 4.1 First Autonomous Batch

**Config focus:** Initial calibration of mask probability, temperature, and acceptance band.

- Start with conservative settings: low mask probability, low temperature, tight acceptance band
- Run 5–10 iterations unattended
- Review batch summary and listening pack

### 4.2 First Listening Check-In

Project owner listens to batch output. Key questions:
- Are variations perceptible at all? (If not: mask probability or editable codebook range too restrictive)
- Do they sound like different hits or like processing artifacts? (If processing: mask schedule or temperature wrong)
- Are transients still clean? (If not: attack masking too aggressive)

Write listening notes. Start supervised session to adjust if code changes are needed.

### 4.3 Subsequent Autonomous Batches

Iterate. Each batch refines based on previous listening notes and metric trends.

**Expected tuning axes (in likely order of importance):**
1. Editable codebook range (6–8 → 5–8 → 4–8 if needed)
2. Mask probabilities (attack vs tail, absolute values)
3. Sampling temperature
4. Acceptance band width
5. Training duration / learning rate

**Expected pattern:** 3–5 autonomous batches with listening check-ins between each, over 2–4 weeks.

### 4.4 Expand to Additional Instrument Families

Once snare variations are calibrated:
- Add kicks, hats, toms, percussion to training data
- Re-run calibration — different instruments may need different settings
- Evaluate whether a single model generalises across families or per-family models are needed

### 4.5 — GATE B: Baseline Calibration

**Gate criteria:** Generated variation magnitude falls within the IQR of real round-robin variation (MR-STFT and MFCC distance) while simultaneously exceeding a minimum variation threshold (not copies). Generated variations must sound better than the procedural baseline from Gate 0 in a direct A/B comparison.

**Gate evaluation:**
1. Generate variations for 50+ held-out samples across instrument families
2. Compute all metrics, compare to baseline distributions
3. Machine-gun proxy test: compare scores to real RR baseline AND procedural baseline
4. Direct A/B listening comparison: ML variations vs procedural variations vs real RR (project owner)
5. Write gate evaluation: `reports/gate-B-evaluation.md`

**Possible outcomes:**
- **Pass:** Both axes satisfied. Metrics in-band. Sounds convincing and better than procedural baseline. Proceed to Phase 5.
- **Adjust:** One axis satisfied but not both. Continue tuning with specific focus.
- **Fail:** Cannot satisfy both axes simultaneously after sustained effort, or cannot beat the procedural baseline. The procedural baseline from Gate 0 becomes V1. Evaluate whether it's sufficient for production use. If not, consider a hybrid approach: procedural variation for gross parameters (gain, micro-timing, transient shape) combined with ML-generated texture variation in the tail only. Document what specifically the ML approach failed at to inform whether a different architecture is worth investigating for V2.

**This is the core feasibility gate. It requires explicit project owner approval.**

### Phase 4 Deliverables
- [ ] Multiple autonomous batches completed with listening check-ins
- [ ] Calibrated hyperparameters for snares
- [ ] Expansion to additional instrument families (if snares pass)
- [ ] Gate B evaluation document written
- [ ] Project owner has listened and approved (or directed pivot)
- [ ] Session log updated

---

## Phase 5: Cross-Library Generalisation

**Goal:** Verify the model generalises to held-out libraries with different recording conditions and to out-of-distribution found sounds. This phase ends at **Gate C**.

### 5.1 Cross-Library Evaluation

- Run inference on the test split (held-out entire libraries)
- Compute full metric suite
- Compare to dev split results — is there significant degradation?
- Listen to cross-library outputs (project owner)

### 5.2 OOD Found-Sound Evaluation

- Run inference on the OOD evaluation pack (metal hits, paper, foley, etc.)
- Evaluate:
  - Similarity to input (must stay high — the model shouldn't destroy unrecognised inputs)
  - Variation magnitude vs real RR baseline (must be comparable)
  - Manual listening

### 5.3 Address Generalisation Gaps

If cross-library or OOD performance degrades:
- Diagnose: is it specific instrument types? Recording conditions? Frequency ranges?
- Options: add more diverse training data, add library/style conditioning, adjust normalisation
- Run targeted autonomous batches if hyperparameter tuning might help

**Three approaches to explore (in order of simplicity):**
1. **Model-inferred variation:** Test whether the model naturally scales variation magnitude to match the instrument's spectral complexity. If kicks and hats already produce family-appropriate variation without per-family tuning, a single universal config may suffice.
2. **Source-adaptive thresholds:** Derive acceptance thresholds from source sample properties (e.g., spectral complexity, transient sharpness). This avoids needing family labels at inference time.
3. **Single universal threshold:** If neither adaptive approach is needed, use one acceptance band for all instruments. This is the simplest operationally and should be the default unless evidence shows it fails.

### 5.4 — GATE C: Cross-Library Generalisation

**Gate criteria:** The model produces convincing variations on held-out libraries and does not destroy OOD inputs.

**Gate evaluation:**
1. Full metrics on test split and OOD pack
2. Side-by-side comparison with dev split performance
3. Listen to cross-library and OOD outputs (project owner)
4. Write gate evaluation: `reports/gate-C-evaluation.md`

**Possible outcomes:**
- **Pass:** Generalisation holds. Proceed to Phase 6.
- **Adjust:** Add training data diversity or conditioning. Re-evaluate.
- **Scope reduction:** Accept library-specific models as valid V1 — the core hypothesis is still validated if the model works within seen recording conditions.

**This gate requires explicit project owner approval.**

### Phase 5 Deliverables
- [ ] Cross-library evaluation complete
- [ ] OOD evaluation complete
- [ ] Generalisation gaps identified and addressed (or scope adjusted)
- [ ] Gate C evaluation document written
- [ ] Project owner has listened and approved
- [ ] Session log updated

---

## Phase 6: Polish & Delivery

**Goal:** Clean up, document, and produce the final V1 deliverable: a reliable pipeline that takes a one-shot drum sample and produces 4–8 variations as 44.1 kHz / 16-bit / stereo WAV files.

### 6.1 Post-Processing Chain Verification

- Verify the full post-processing chain produces correct output format
- DC offset removal, level matching, tail fade, dither — all working correctly
- Spot-check 50+ outputs: correct sample rate, bit depth, channel count, no clipping, no DC offset

### 6.2 End-to-End Pipeline

- Single command or script: input WAV → output N variation WAVs
- Handles arbitrary input length (truncates to 1.0s with appropriate fade)
- Handles mono input (converts to stereo if needed)
- Clear error messages for unsupported formats
- Configurable: number of variations, acceptance strictness

### 6.3 Final Evaluation

- Run the full evaluation suite on a comprehensive set of inputs
- Produce a final report: `reports/v1-final-evaluation.md`
- Include: metric summary, baseline comparison, listening assessment, known limitations

### 6.4 Documentation

- Update CLAUDE.md with any new conventions or mistakes learned
- Write a brief usage guide in the README
- Document the model's known strengths and limitations
- Archive experiment history (W&B links, key reports)

### Phase 6 Deliverables
- [ ] Post-processing chain verified
- [ ] End-to-end pipeline working with single command
- [ ] Final evaluation report written
- [ ] Documentation complete
- [ ] V1 complete: the system reliably produces 4–8 variations that pass the machine-gun test

---

## Appendix: Key Metrics Reference

Quick reference for metrics used across phases.

| Metric | What It Measures | Target |
|---|---|---|
| MR-STFT distance (to input) | Spectral similarity to source | Within real RR IQR |
| MR-STFT distance (attack window) | Transient preservation | Close to 0 (minimal change) |
| MFCC distance (to input) | Timbral similarity | Within real RR IQR |
| Token change rate | Sparsity of edits | Within real RR token change distribution |
| Inter-variation MR-STFT | Diversity between variations | Non-zero, within real RR range |
| Machine-gun proxy score | Perceptual repetition | Comparable to real RR sets |
| Acceptance rate | Generation efficiency | >20% target |
| Attack smear score | Transient integrity | ~1.0 (no smearing) |

## Appendix: Automation Config Template

```json
{
  "batch_id": "batch-001",
  "model_checkpoint": "checkpoints/latest.pt",
  "max_iterations": 10,
  "stagnation_limit": 3,

  "hyperparameters": {
    "mask_p_attack": 0.02,
    "mask_p_tail": 0.08,
    "editable_codebooks": [6, 7, 8],
    "temperature": 0.9,
    "top_p": 0.95,
    "k_candidates": 8,
    "acceptance_band_low": 0.05,
    "acceptance_band_high": 0.25,
    "learning_rate": 2e-4,
    "batch_size": 64,
    "attack_frames": 2
  },

  "guardrails": {
    "rollback_thresholds": {
      "mrstft_to_input_p95": 0.30,
      "attack_smear_score_mean": 0.10,
      "acceptance_rate_min": 0.10
    }
  },

  "eval": {
    "dev_samples": 50,
    "test_samples": 0,
    "ood_samples": 0,
    "generate_listening_pack": true,
    "generate_machine_gun_test": true
  }
}
```
