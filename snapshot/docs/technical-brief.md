# VARIATION SAMPLER — Technical Brief

## 1. Project Overview

Variation Sampler is a machine learning system that generates perceptually convincing round-robin variations from a single one-shot drum sample. Given one recording of a drum hit, it produces 4–8 additional versions that sound like the same instrument struck again under the same conditions — not like the same recording replayed, and not like the original sample has been processed.

The project solves the "machine-gun effect": rapid retriggering of an identical sample sounds obviously artificial because human hearing is extraordinarily sensitive to exact repetition. Professional sample libraries solve this with round-robin sampling (multiple recordings of the same hit), but that requires expensive studio sessions. Variation Sampler generates equivalent results from a single source sample using a model trained on thousands of real round-robin sets.

**V1 is a proof of concept.** Its sole purpose is to answer: can an ML model generate drum sample variations that preserve source identity while breaking perceptual repetition? Everything else — tooling, distribution, velocity layers — is contingent on a clear yes.

### Audience and Evaluator

V1 has an audience of one: the project owner, a musician who works extensively with drum samples and has direct daily experience of the machine-gun limitation. The project owner is both developer and evaluator. Perceptual judgement from this trained listener is the ultimate authority — it overrides all quantitative metrics.

### What This Is Not

- Not a drum synthesizer or physical model
- Not a velocity layer generator (that's a fundamentally different problem — synthesis, not variation)
- Not a real-time performance tool
- Not a product with users, UI, or distribution

The model learns statistical patterns of how round-robin recordings differ from each other. It does not model drum physics. From thousands of examples of "here are five recordings of the same snare hit — notice how they differ," it learns a general template of what "same but different" means for percussive audio, then applies that template to unseen one-shots.

---

## 2. Definition of Done

V1 is complete when all of the following are satisfied:

**The machine-gun test.** Generated variations, played in rapid sequence (16th notes at 120 BPM), sound like the same drum struck multiple times — not like the same recording replayed. Evaluated by expert listening from the project owner. This takes precedence over any quantitative metric.

**Two-axis validation.** Variations must simultaneously satisfy:
- *Similarity:* Each variation is perceptibly the same instrument — same pitch, timbre, decay profile, energy envelope. Does not sound like a different drum or like the source has been processed.
- *Variation:* Each variation is perceptibly different from the input and from every other variation. Magnitude of difference falls within the range observed in real round-robin recordings, as measured by multi-resolution spectral distance and MFCC distance against a calibrated baseline.

**Cross-source validation.** The above criteria are met on held-out libraries with different recording conditions, not only on libraries in training data. The model has learned variation patterns, not studio fingerprints.

**Concrete output.** Given a single one-shot drum sample, the system reliably produces 4–8 variations as 44.1 kHz / 16-bit / stereo WAV files that pass the above tests.

---

## 3. Scope Boundaries

### In Scope for V1
- Round-robin variation generation from a single one-shot drum sample
- Output: 4–8 variations per input as 44.1 kHz / 16-bit / stereo WAV files
- Training on existing commercial round-robin drum libraries
- Cloud-based training and inference on Google Cloud
- Starting with snares, expanding to other drum families if initial results are promising
- Automated iteration loop with quantitative metrics and periodic human listening evaluation
- Go/no-go gates for early, cheap failure

### Explicitly Out of Scope for V1
- Velocity layer generation (fundamentally different problem — synthesis, not variation)
- Ableton instrument packaging (existing tooling handles this independently)
- Real-time synthesis or live performance
- Any user-facing application, interface, or distribution
- Cross-domain generalisation to non-percussive audio
- Product polish, optimisation, or deployment concerns

---

## 4. Design Principles

**Perceptual quality is the ultimate authority.** Metrics serve the ear, never the other way around. If a variation measures well on spectral distance metrics but sounds wrong to a trained listener, it is wrong. Every technical decision is subordinate to how the output sounds in actual musical use.

**Feasibility before everything.** Speed-to-answer matters more than polish, elegance, or completeness. The project is structured to reach a clear yes or no as cheaply and quickly as possible.

**Subtlety is the product.** Variations must not sound like different drums, or like the original has been processed. They must sound like the same performer hit the same instrument again. The target is the narrow perceptual band where real round-robin recordings naturally sit — different enough to break repetition, similar enough to preserve identity.

**Narrow and concrete over broad and speculative.** The ambitious long-term vision earns its place only after the core hypothesis is validated. Scope discipline is how the project avoids spending months building infrastructure for something that might not work.

---

## 5. Procedural Baseline (Pre-Gate 0)

Before training any model, build a simple procedural variation generator as a quality floor. Apply per-variation randomised: gain (±0.3 dB), pitch (±3 cents), transient shaping (subtle attack time jitter), short convolution with micro-impulses for subtle tonal colouring, and sample-level micro-timing offset (±5 samples).

Spend one weekend building this, render machine-gun tests across snares/kicks/hats, and do a serious listening session. Document what it gets right and where it falls apart. This becomes the concrete baseline every ML output is compared against — not "does it sound good?" but "does it sound better than the cheap version?"

If the procedural baseline is already good enough for the project owner's use cases, the ML approach is unnecessary. If it fails in specific, articulable ways (e.g., "timbral variation is always in the wrong dimension — it sounds like EQ, not like a different strike"), those failure modes become the ML model's explicit targets.

**Gate 0: Procedural baseline evaluated. Failure modes documented. ML approach justified or project complete.**

---

## 6. Architecture — Settled Decisions

Two rounds of research converged on a specific technical approach. These are settled for V1 unless empirical results force a pivot.

### 6.1 Codec: DAC (Descript Audio Codec)

**Configuration:** 44.1 kHz full-bandwidth, frozen (do not fine-tune unless reconstruction quality on source material is demonstrably the bottleneck).

| Parameter | Value |
|---|---|
| Sample rate | 44.1 kHz |
| RVQ codebooks | 9 |
| Codebook size | 1024 entries (10 bits/token) |
| Stride | 512 samples |
| Latent frame rate | 86 Hz |
| Token throughput | 774 tokens/sec (9 x 86) |
| Stereo support | Yes |
| Codegram shape (1s audio) | `[9, 86]` integer tensor, values 0–1023 |

**Why DAC over alternatives:**
- Natively targets 44.1 kHz (the delivery spec) — no resampling needed
- Universal codec designed across speech/environment/music domains
- Open-source with released pretrained weights for 44.1k/24k/16k, mono/stereo
- EnCodec's stereo model is 48 kHz (requires resampling) and uses chunk-based encoding with per-chunk scale factors that complicate short one-shot processing
- SoundStream lacks maintained pretrained weights at 44.1 kHz

**Freeze rationale:** Codec representations can be fragile under fine-tuning (catastrophic forgetting, idempotence degradation). Fine-tuning should only be considered if DAC reconstruction is proven to be the quality bottleneck, and then conservatively (adapters/LoRA + data replay, not full fine-tuning).

### 6.2 Variation Model: Masked-Token Micro-Inpainting Transformer

The model does not generate full new codegrams. It copies most tokens from the input and resamples a small, stochastic subset — primarily in later RVQ codebooks that encode fine residual detail.

**Core insight:** Codec token spaces are not perceptually smooth. Small changes in token index do not reliably produce small changes in perceived audio, because codebook indices are categorical (entry 123 is not inherently "closer to" entry 124 than to entry 900). This means "add a small delta in latent space" would likely produce synthetic-sounding results. The micro-inpainting approach sidesteps this entirely — instead of asking "what small change produces a subtle variation?", it asks "what tokens would plausibly appear here if this were a different recording of the same hit?" The "smallness prior" is structural (sparse edits in late codebooks) rather than numeric.

**Editable codebook range — probabilistic gradient, not hard cutoff:** Allow edits across codebooks 3–8, but with exponentially decaying mask probability for earlier codebooks. For example: codebook 3 at ~0.5% edit chance, codebook 4 at ~1%, codebook 5 at ~2%, codebooks 6–8 at ~8–15%. This is more realistic than a hard cutoff at codebook 6, because real round-robin variation includes subtle changes in spectral envelope and harmonic balance (encoded in earlier codebooks), not just fine texture (late codebooks). The acceptance filter remains the ultimate guardrail — if an earlier-codebook edit produces too large a perceptual change, it gets rejected. Start conservatively (low probabilities across 4–8), and tune the gradient based on listening.

**Cross-codebook coherence:** DAC's RVQ structure means each codebook encodes the residual after all previous codebooks. Changing a token in codebook 7 while holding codebooks 0–6 fixed is valid because the decoder sums independent codebook embeddings — there is no autoregressive dependency between codebooks at decode time. However, the *training distribution* of codebook 7 tokens is conditioned on codebooks 0–6 being correct. Verify that the model's predicted tokens for later codebooks remain in-distribution when conditioned on frozen earlier codebooks. If generated tokens cluster in unusual regions of the codebook embedding space, this signals a distribution mismatch.

**Training objective:** Masked token cross-entropy on round-robin pairs. Sample pairs (A, B) from a round-robin set, encode both with DAC, train the model to predict masked subsets of z_B conditioned on z_A. Across many pairings, maximum likelihood training approximates the conditional distribution of plausible RR variations.

**No need to backpropagate through DAC decoder** for token cross-entropy training. The codec stays frozen and the training loss is purely categorical CE over target tokens.

### 6.3 Quality Control: Audio-Domain Acceptance Filtering

Decode every candidate variation, compute perceptual distance to input, reject anything outside a calibrated band derived from real round-robin data. This is the primary guardrail against synthetic-sounding output.

The acceptance loop: generate K candidates → decode with DAC → compute perceptual distance (MR-STFT and/or MFCC distance) → keep those within the target band → repeat if needed.

**Acceptance rate matters:** If below ~20%, inference is expensive and slow (generating 5x more candidates than kept). Sustained low rates signal the model is poorly calibrated (mask rate or temperature too high) or the acceptance band is too narrow. Track acceptance rate as a first-class metric.

### 6.4 Training Framework and Cloud

- **Framework:** PyTorch (ecosystem support — Diffusers, AudioCraft, DAC repo are all PyTorch-first)
- **Training GPU:** A100 (A2 series) on Google Cloud, ~$3.67/hr on-demand
- **Inference/prototyping GPU:** L4 (G2 series) ~$0.85–1.36/hr, or T4 ~$0.35/hr as budget fallback
- **Model registry:** Vertex AI Model Registry for version tracking
- **Experiment tracking:** Weights & Biases (training curves, metric comparisons, inline audio sample playback)

---

## 7. Model Architecture Detail

### 7.1 Input/Output Specification

**Inputs:**
- `z_in`: int tensor `[B, 9, T_max]` (values 0–1023) — DAC codegram of input one-shot
- `mask`: boolean tensor `[B, Nq_edit, T_max]` — which positions are eligible for editing
- `Nq_edit` ∈ {1..6}, e.g., codebooks 3–8 means Nq_edit=6 (with per-codebook probability gradient)

**Stereo encoding:** DAC encodes stereo as a joint representation — both channels are folded into the same codegram. The transformer operates on this joint representation without explicit channel separation. This means stereo image is implicitly preserved by the frozen codec's reconstruction, not explicitly modelled. Verify empirically that DAC round-trip preserves stereo image on drum one-shots before training. If stereo reconstruction is lossy on transients, consider mono processing with stereo image restored post-generation via the original's mid/side balance.

**Model internals:**
- Embedding tables `E_i: [1024] → R^D` for each codebook i
- Time-step representation: `h_t = Σ_i E_i(z_in[i,t]) + E_mask(masked)` (sum embeddings across codebooks per frame — this is the approach used by TRIA)
- Small bidirectional transformer encoder over `t=1..T_max`
- Per editable codebook i, a projection head to logits: `[B, T_max, 1024]`

**Outputs:**
- Logits `L_i[t]` for each editable codebook at each time step
- Sampled tokens for masked positions via categorical sampling with temperature

### 7.2 Inference Flow (Pseudocode)

```
# Constants for DAC 44.1k fullband
NQ = 9
CODEBOOK_SIZE = 1024
FPS = 86
T_MAX = 86              # 1.0s
EDIT_CODEBOOKS = [3,4,5,6,7,8]
ATTACK_FRAMES = 2       # ~23ms at 86 Hz

# Per-codebook mask probability multiplier (probabilistic gradient)
CB_MASK_MULT = {3: 0.06, 4: 0.12, 5: 0.25, 6: 1.0, 7: 1.0, 8: 1.0}

# Encode input
z_in = dac.encode(wav_in)               # [NQ, T]
z_in = pad_or_trim(z_in, T_MAX)

# Build stochastic edit mask with per-codebook gradient
mask = zeros([len(EDIT_CODEBOOKS), T_MAX])
for cb_idx in EDIT_CODEBOOKS:
  for t in range(T_MAX):
     p_base = p_attack if t < ATTACK_FRAMES else p_tail
     p = p_base * CB_MASK_MULT[cb_idx]
     mask[cb_idx, t] = Bernoulli(p)

# Model forward pass
logits = variation_model(z_in, mask)     # per editable cb: [T_MAX, 1024]

# Sample new tokens only where masked
z_out = z_in.clone()
for cb in EDIT_CODEBOOKS:
  for t in masked_positions(cb):
     z_out[cb, t] = categorical_sample(logits[cb, t], temperature)

wav_out = dac.decode(z_out)

# Acceptance filter
if perceptual_distance(wav_out, wav_in) not in target_band:
    repeat sampling (up to K tries) or lower temperature
return wav_out
```

### 7.3 Smallness Prior — Three Complementary Mechanisms

1. **Structural smallness: restrict where edits are allowed**
   - Codebook restriction: only edit the last Nq_edit codebooks
   - Time restriction: lower mask probability on attack frames (first ~23ms) vs tail frames

2. **Statistical smallness: penalize edit rate**
   - `change_rate = mean(z_out != z_in)` on editable codebooks
   - Penalty: `L_small = λ * |change_rate - target_change_rate|`
   - `target_change_rate` derived from real round-robin pairs

3. **Perceptual smallness: acceptance filtering**
   - Decode with DAC and compute perceptual distance
   - Hard reject anything outside the target band (calibrated from real RR data)

### 7.4 Stochasticity Sources

Variation comes from two places:
- **Random masking pattern** — which tokens are eligible to change
- **Sampling noise** — temperature/top-p when choosing replacement tokens from logits

Calibration: precompute a distribution of real round-robin variation magnitude (audio-domain and token-domain) from training libraries. Tune mask_prob, editable codebook range, and sampling temperature until generated variations match the median and IQR of that real distribution.

### 7.5 Curriculum Strategy

- **Phase 1:** Probabilistic gradient across codebooks 6–8 only; very low mask probabilities; tight acceptance threshold
- **Phase 2:** Open gradient to codebooks 4–8; increase mask probabilities; acceptance band widened to match real RR baseline
- **Phase 3 (if needed):** Open gradient to codebooks 3–8; tune per-codebook multipliers based on listening

---

## 8. Data Pipeline

### 8.1 Training Data Scale

| Tier | Round-Robin Sets | Pairs (N=5) | Scope |
|---|---|---|---|
| Minimum viable (POC) | 500–1,500 | 10k–30k | 1–2 instrument families (snares) |
| Recommended | 5,000–15,000 | 100k–500k | Kicks/snares/hats/percussion |
| Ideal | 20,000+ | 400k+ | Multi-library, multi-mic diversity |

One training example = one articulation, one velocity bin, one instrument/mic setup, N round-robin hits (typically 3–8 in commercial libraries). Using ordered pairs (input → target), each set of N yields N*(N-1) training pairs.

The project owner has access to extensive commercial deeply-sampled libraries, so data availability should not be a bottleneck.

### 8.2 Preprocessing Pipeline (In Order)

1. **Import** WAV files from commercial libraries, organised by library/kit/instrument/articulation/velocity bin
2. **Trim silence** with conservative margins — never clip pre-transient noise (it's part of the realism signature)
3. **Transient-align** using:
   - Coarse onset estimate via spectral flux onset strength (librosa `onset_strength` + `onset_detect`)
   - Backtrack to pre-onset reference (librosa `onset_backtrack`) for consistent alignment
   - Fine alignment via short-window cross-correlation on the attack band (~5–20ms window, optionally high-passed)
   - Align to onset point, not absolute peak (standardises "when the event begins" not "when maximum happens")
4. **Pad or truncate to fixed length: 1.0 second for V1.** This is a snare-focused simplification. 1.0s comfortably contains snare and most kick samples with natural decay. It is insufficient for cymbals (rides, crashes), long toms, and many found sounds. When expanding to other drum families, this must become instrument-family-dependent (e.g., 0.5s for closed hi-hats, 2.0s for rides/crashes) or variable-length with appropriate masking. For V1, restrict evaluation claims to instruments whose natural duration fits within 1.0s.
5. **Loudness normalise** within ±1 dB tolerance — reduces level as a confound without destroying natural micro-dynamics. Do not over-normalise.
6. **Encode through DAC** and cache resulting codegrams as tensors on disk (this is critical for iteration speed — never re-encode on each training run)
7. **Compute and cache ground-truth baseline metrics**: within-set pairwise distances (MR-STFT, MFCC/MCD) for every round-robin set, aggregated into distributions by instrument family

**Key libraries:** librosa (onset detection, spectral analysis), madmom (alternative onset detection)

### 8.3 Velocity Handling

**Exclude velocity layer variation from V1.** Keep training pairs strictly within the same velocity bin. Do not include cross-velocity pairs. The model should learn micro-variation at constant perceived intensity, not dynamics.

### 8.4 Augmentation

**Safer augmentations (small, controlled):**
- Very small gain perturbations (sub-dB scale)
- Very small micro-time shifts (after alignment) as robustness training, not as "variation ground truth"

**Risky augmentations (avoid — they corrupt the learning target):**
- Pitch shifting / time stretching beyond extremely minor ranges
- Heavy convolution reverb / re-amping (moves model toward "room variation" not "hit variation")

**Bandwidth integrity warning:** DAC can fail to reconstruct high frequencies if trained on mixed-bandwidth sources. Drums rely heavily on high-frequency transient detail. Ensure training data is full-bandwidth 44.1 kHz.

### 8.5 Splitting Protocol

Split by group key: `(library_id, kit_id, instrument_id, articulation_id, mic_perspective_id)`. No group key may appear in more than one split.

| Split | Purpose | Composition |
|---|---|---|
| Train | Model training | Bulk of data |
| Dev | Fast iteration metrics | Same libraries, different kits/instruments |
| Test | Generalisation testing | Hold out entire libraries (different rooms/mics/preamps) |
| OOD eval pack | Found-sound robustness | ~50–200 one-shot found sounds (metal, paper, plastic, foley) with no RR ground truth |

For OOD evaluation: measure similarity-to-input (must stay high) and variation magnitude vs real RR baseline (must be comparable), plus manual listening.

Without round-robin ground truth, OOD evaluation is purely subjective. Similarity-to-input and variation magnitude metrics provide sanity checks (the output hasn't diverged wildly or collapsed to a copy), but they cannot validate that the variation sounds like a plausible re-strike rather than a processing artifact. OOD evaluation is a listening-only gate. Document this clearly in results: "passed OOD listening check" is a weaker claim than "matched real RR baseline on held-out library."

**Found-sound generalisation is explicitly deferred.** V1 trains on commercial drum libraries only. The OOD eval pack tests basic robustness (the model shouldn't destroy unfamiliar inputs), but V1 makes no claim about generating convincing variations of non-drum found sounds. That capability, if achievable, requires its own training data and research phase.

---

## 9. Evaluation Metrics

### 9.1 Similarity Metrics (Is It the Same Drum?)

**Multi-resolution STFT distance** (spectral convergence + log magnitude):
- Compute on a focused window around the transient (first 30–80ms) AND on the full clip
- Three-resolution setup: 10ms/2ms hop, 25ms/5ms hop, 50ms/10ms hop (FFT sizes 512, 1024, 2048)
- The shortest window protects attack sharpness; longer windows stabilise pitch/tonality

**MFCC distance / Mel-Cepstral Distortion (MCD):**
- Timbre drift detector
- DTW usually unnecessary with good transient alignment

**Perceptual metrics (secondary — for ranking and artifact detection, not primary optimisation):**
- ViSQOLAudio: full-reference audio quality metric beyond narrowband speech
- CDPAM: learned perceptual similarity metric trained on human judgments, differentiable
- PESQ: speech-oriented, may be poorly matched to drums, but useful as gross artifact alarm

**Phase-aware metric (for layering safety):**
- Multi-resolution complex STFT distance (evaluates real and imaginary parts, not just magnitude) — penalises phase shifts in fundamental frequencies that would cause cancellation when layering
- This is secondary to magnitude-based metrics but important for production use where drum samples are frequently stacked

### 9.2 Variation Metrics (Is It Actually Different?)

**Inter-variation distance distribution:** Pairwise distances between generated variations using same metrics as similarity. Target: non-zero difference with bounded spread.

**Ground-truth scale matching:** For each real RR set, compute pairwise hit distances, aggregate by instrument family → distribution (median/IQR) of natural variation magnitude. Generated variations must fall within this range.

**Token change rate:** `mean(z_out != z_in)` on editable codebooks. Compare to real RR token change rate distribution.

### 9.3 Machine-Gun Proxy Score

No standardised metric exists. Practical approach:

1. Render 8-hit sequence at 16th notes, 120 BPM
2. Extract per-hit feature vector (MFCCs over first 50–100ms + envelope/centroid/flux)
3. Compute self-similarity matrix over the 8 events
4. Score: `MachineGunScore = mean(cosine_similarity(feature_i, feature_j)) for i≠j`
   - Higher = everything identical (bad)
   - Lower = more varied (good)
   - Must be constrained by identity metrics to avoid rewarding wild divergence

### 9.4 Anti-Gaming Metrics

Specific metrics to catch Goodhart's Law failure modes:

**High-frequency energy delta:** Calculate energy above 12 kHz in the input and each variation. Hard-reject any variation that increases HF energy beyond a small threshold. This catches the "add inaudible HF noise to pass spectral distance metrics" failure mode.

**Transient cross-correlation:** After generation, compute cross-correlation of the attack region (first 5–10ms) between input and variation. Score should be high (>0.95). This catches phase shifts introduced by the DAC decoder that would cause cancellation when layering.

### 9.5 Goodhart's Law — Concrete Failure Modes to Watch For

Metrics can be gamed. Watch for:
- **Broadband noise addition:** Technically different by spectral metrics, within distance band, but sounds like tape hiss not a different hit
- **Timing shifts in tail/decay:** Metrically distinct, perceptually identical or worse (phase cancellation when layering)
- **Inaudible high-frequency changes:** Spectrally different above 15 kHz, completely imperceptible — caught by HF energy delta metric
- **Model that only varies level:** Passes distance metrics but not doing real timbral variation

**Primary defence:** Periodic listening check-ins by the project owner. No metric suite substitutes for a trained ear evaluating "different hit" vs "same hit with artifacts."

---

## 10. Post-Processing Chain

Applied to DAC decoder output before delivery as final WAV:

1. **Transient phase alignment** — Cross-correlate the attack region (~first 5ms) of the variation with the input; apply sub-sample shift to re-align. This corrects any micro-timing shift introduced by DAC decoding, ensuring the variation remains phase-coherent with the input for layering.
2. **DC offset removal** — subtract mean per channel
3. **Level matching** — match RMS over fixed window after onset (10–200ms) to input sample, clamp ±1 dB. Do NOT peak-normalise independently per variation (destroys natural level variation)
4. **Tail fade-out** — short equal-power fade (5–20ms) only when hard truncation occurred
5. **Dither to 16-bit PCM** — TPDF dither, applied as final step after all other processing
6. **Export** — 44.1 kHz / 16-bit / stereo WAV

---

## 11. Iteration Approach

### 11.1 Pre-Gate A: Manual Iteration

Before Gate A, iteration is manual. Train a run, generate samples, listen, adjust hyperparameters in a config file, repeat. Track results in W&B. The goal is to understand the parameter space and reach first listenable outputs as quickly as possible without building automation infrastructure.

This is consistent with "feasibility before everything" — answering "does this work at all?" doesn't require an automated loop.

### 11.2 Post-Gate A: Automated Iteration Loop

After Gate A confirms the approach can produce clean outputs, build the automated loop. Its value is allowing overnight/weekend runs that cover parameter space while the project owner is unavailable. The loop is deliberately simple: hyperparameter-only edits, no code changes. **If building the loop takes more than 2 days, defer further and continue manual iteration.**

**Loop architecture** — a Python script running on a GPU VM:

1. **Train** (or continue training) with current config
2. **Generate** test samples from held-out inputs
3. **Compute** all metrics
4. **Write** structured JSON report
5. **Send** JSON report to Claude via the API (standard messages API call, not Claude Code)
6. **Claude returns** a new config JSON containing only hyperparameter adjustments
7. **Runner script** applies new config and starts next iteration

**What Claude may change:** mask probabilities (attack/tail, per-codebook multipliers), temperature, top_p, k_candidates, acceptance band bounds, learning rate, batch size, attack frame count. Nothing else.

**What Claude must provide:** A `reasoning` field explaining why each change is proposed, before the `proposed_changes`. This reduces erratic hyperparameter leaps.

**Guardrails:**
- **Auto-rollback** if key metrics regress past thresholds
- **Hard stop** after N iterations without improvement (default: 3)
- **Hard stop** after M total unattended iterations (default: 10)
- **Maximum step size** per parameter per iteration (e.g., temperature can change by at most ±0.1, mask probability by at most ±0.02) — prevents wild jumps into noise-generating territory

### 11.3 Listening Check-In Cadence

Let the automated loop run 5–10 iterations unattended, then spend 20–30 minutes listening. Focus on:
- Does the best output sound like a different hit, or like a processed version of the same hit?
- Does the worst output reveal a systematic failure mode (transient smearing, tonal artifacts)?
- Are variations "interesting" the way real round-robins are, or do they feel mechanical?

After listening, write a brief natural-language note summarising what you heard. This gets appended to the next Claude API prompt alongside the metrics JSON. Ears overrule metrics.

### 11.4 Code-Level Changes

Architecture changes and code modifications happen between automated run batches, not within them. These are done in human-supervised Claude Code sessions, triggered by findings from listening check-ins.

---

## 12. Go/No-Go Gates

The project is designed to fail fast. Four gates, in order:

### Gate 0: Procedural Baseline (Target: 1 weekend)

Build the procedural variation generator (Section 5). Listen critically. Document failure modes.

**Decision:** If the procedural baseline is good enough for production use, the ML approach is unnecessary — project complete. If it fails in articulable ways, those failure modes become the ML model's targets. Proceed to Gate A.

### Gate A: First Listenable Outputs (Target: 2–3 weeks after Gate 0)

Generated variations must preserve transient integrity — no obvious ringing, smeared attacks, or tonal artifacts on snares and kicks. If the probabilistic codebook gradient with acceptance filtering cannot produce clean transients, the approach may be fundamentally fighting codec or manifold issues.

**Decision:** Continue, adjust codebook range, or pivot.

### Gate B: Baseline Calibration (Target: 4–6 weeks)

Generated variation magnitude must fall within the IQR of real round-robin variation (MR-STFT and MFCC distance), while simultaneously exceeding a minimum variation threshold (no copies). Generated variations must sound better than the procedural baseline from Gate 0 in a direct A/B comparison.

**Decision:** If both axes cannot be satisfied simultaneously, pivot.

**If Gate B fails:** The procedural baseline from Gate 0 becomes V1. Evaluate whether it's sufficient for the project owner's actual production use. If not, consider a hybrid approach: procedural variation for gross parameters (gain, micro-timing, transient shape) combined with ML-generated texture variation in the tail only — a reduced-ambition version that may still beat pure procedural. Document what specifically the ML approach failed at to inform whether a different architecture (e.g., diffusion in continuous latent space rather than discrete token inpainting) is worth investigating for V2.

This is the core feasibility gate.

### Gate C: Cross-Library Generalisation (Target: 6–8 weeks)

Test on held-out libraries with different recording conditions. If generalisation collapses, the model has learned studio fingerprints rather than variation patterns.

**Decision:** Add more diverse training data, add conditioning/normalisation, or accept library-specific models as a valid V1 scope reduction.

---

## 13. Cloud Cost Estimate

Order-of-magnitude estimates for the full POC phase:

| Item | Cost |
|---|---|
| Data preparation and DAC encoding | Mostly CPU, minimal GPU cost |
| Training + sweeps (~10 runs x 8 hrs on A100 @ ~$3.67/hr) | ~$300 |
| Inference/generation for test samples | Minimal (same VM) |
| Early dev on L4/T4 | Reduces cost at expense of speed |
| **Total POC budget** | **~$300–$600 compute** + storage/egress |

Claude API costs for the automated loop depend on prompt length and iteration count but should be modest relative to GPU costs.

---

## 14. Key Prior Art

Ranked by implementation value:

**TRIA (2025)** — Masked-token transformer for drum audio generation using DAC codegrams. Closest existing system. Its tokenisation (sum codebook embeddings per frame), SoundStorm-style coarse-to-fine unmasking, and DAC integration are directly reusable. Differs in using rhythm conditioning rather than input-hit conditioning.

**VampNet (2023)** — Masked acoustic token modeling for music variation using bidirectional transformer. The prompting/sampling design (multiple sampling passes, temperature control) is directly relevant to controlling variation magnitude.

**DAC (Descript Audio Codec)** — The chosen codec. 44.1 kHz native, stereo, open-source with pretrained weights. Universal design across speech/environment/music.

**Real-time Timbre Remapping with DDSP (2024)** — Uses snare drum performance variation as a case study. Supports the premise that subtle micro-variation is musically meaningful and learnable. Offers engineering ideas for onset-anchored processing and feature difference losses.

**ICLR 2026 discrete diffusion for token inpainting** — Token-based audio inpainting via discrete diffusion. Targets restoration, but the mechanism (structured corruption + regularization over discrete tokens) is compatible with micro-inpainting for variation.

**No published work directly claims "generate round-robin sets from a single one-shot" as its primary contribution.** The prior art is adjacent, not on-the-nose. This increases the importance of the empirical go/no-go gates.

---

## 15. Confidence Assessment

| Dimension | Confidence | Notes |
|---|---|---|
| Generate "different" outputs | High | Multiple sampling seeds in a generative model almost guarantees diversity |
| Outputs remain "same source" while being "different enough" | Medium | This is the hard part. Micro-inpainting + acceptance filtering is the most credible path, but untested for this specific application |
| Consistently beat hand-designed humanisation/randomisation | Medium-low | Artifacts are easy to introduce at the transient; "subtle but natural" is a high bar |

The project is designed around this uncertainty. Gates are early, scope is narrow, failure criteria are concrete. This is a genuine experiment, not an implementation of a proven technique.

---

## 16. Key Technical Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic-sounding variation (ringing, grain, pre-echo) | Acceptance filtering on decoded audio; periodic listening check-ins |
| Category failures (works on kicks, fails on hi-hats) | Category-stratified baselines; test early across families |
| Data leakage / overfitting to specific libraries | Split by group key; hold out entire libraries for test |
| Metric gaming (Goodhart's Law) | Ground-truth scale matching + mandatory listening gates |
| Late-codebook-only edits too conservative | Tuning axis: cautiously open earlier codebooks with lower mask probability |
| Low acceptance rate making inference expensive | Track as first-class metric; adjust generation parameters, not just absorb cost |
| Codec reconstruction artifacts on off-distribution inputs | Empirically verify DAC reconstruction on intended input types before committing |

---

## 17. What Success Enables (Deferred)

If V1 validates the core hypothesis:

- **Velocity layer generation (V2):** Training the model to generate variations at different strike velocities — turning a single sample into a dynamically playable instrument. Fundamentally harder (synthesis, not variation) — separate research phase.
- **Instant instrument creation:** Combined with existing Ableton Sampler instrument tooling, turn a few field recordings into a fully playable, velocity-layered, round-robin-equipped virtual instrument in minutes.
- **Democratised deep sampling:** Any sound source becomes a candidate for a production-quality sampled instrument. Note: this requires training on found-sound/foley data, not just commercial drum libraries. V1 does not attempt this.

These are contingent on V1 success and are explicitly deferred.

---

## 18. Ethical Considerations

**Training data:** V1 trains on commercial drum libraries the project owner has legitimately purchased. The model learns statistical patterns of variation, not specific sounds. Output is novel variation of user-provided input, not reproduction of training samples. Analogous to how a session drummer's technique is informed by every drummer they've listened to.

**Memorisation guard:** The trained model must not memorise or regurgitate specific commercial samples. Acceptance filtering and the micro-inpainting architecture (which edits the user's input rather than generating from scratch) inherently mitigate this.

**Commercial implications:** If successful and eventually distributed, this technology could reduce demand for deeply-sampled commercial drum libraries. The counterargument is that it could expand the market by making sample-based production more accessible and creating demand for high-quality source samples as inputs. For V1 (personal use only), this is noted but not actionable.

**Transparency:** If ever distributed, the tool should be clear that variations are ML-generated, not recorded.

---

## Note on the Mental Model

The vision statement uses a DNA/siblings metaphor: round-robin sets are like siblings sharing genetic code but expressing it differently. This is an accurate description of the *perceptual target* — what the output should sound like. However, the model does not literally generate from a shared generative origin the way DNA works. It learns what the *output distribution* of natural variation looks like and reproduces that distribution by making targeted edits to the input. The sibling metaphor describes the perceptual goal, not the mechanism.
