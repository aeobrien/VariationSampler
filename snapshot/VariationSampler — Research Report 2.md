# Variation Sampler Follow‑Up Report: Codec‑Latent Round‑Robin Generation at Implementation Depth

## Codec recommendation and latent space specification

A practical v1 codec choice is the **44.1 kHz universal Descript Audio Codec (DAC)** from entity["company","Descript","audio software company"], used at its full‑bandwidth configuration (≈8 kbps) with **9 residual VQ codebooks**, **1024 entries per codebook**, a **striding factor of 512**, and a **latent frame rate of 86 Hz** for 44.1 kHz audio. citeturn1view0turn4view1turn2view0

This choice is implementation-friendly for the project constraints because it **natively targets 44.1 kHz** (your delivery spec), and the official repository states support for **44.1k/24k/16k and mono/stereo**, with released weights. citeturn1view0

A concise comparison to the two closest alternatives:

- **DAC (recommended)**: 44.1 kHz native; designed as a *universal* codec across domains; in the paper’s baseline table the 44.1 kHz model is configured with **86 latent steps/sec** and **9×10‑bit codebooks** (1024 entries). citeturn2view0turn4view1  
- **EnCodec** from entity["company","Meta","tech company"]: supports **24 kHz mono** and **48 kHz stereo**, with **75 latent steps/sec at 24 kHz** and **150 at 48 kHz**; uses up to **32 codebooks (24 kHz)** or **16 codebooks (48 kHz)**, each 1024 entries. citeturn1view1turn1view2turn4view0turn3search17  
  - Practical drawback for you: 48 kHz stereo is not 44.1 kHz; you’d either resample in/out (adds complexity) or accept a mismatch. Also, the EnCodec repo notes the 48 kHz model encodes in 1‑second chunks with overlap and outputs a per‑chunk scale factor, which you’d need to handle carefully for short one‑shots to avoid unintended level behavior. citeturn1view1turn4view0  
- **SoundStream** from entity["company","Google","tech company"]: evaluated in the original paper at **24 kHz** and uses an encoder/RVQ/decoder design, but it is (a) not natively 44.1 kHz, and (b) in practice less “drop‑in” today than the actively maintained DAC/EnCodec repos for pretrained weights and tooling. citeturn1view3turn2view0  

### Latent/token space structure you should design around (DAC 44.1k)

For **DAC 44.1 kHz @ full bandwidth** (as described in the DAC paper table):

- **Codebooks (RVQ stages)**: `Nq = 9`  
- **Codebook size**: `V = 1024` entries (10 bits/token)  
- **Frame rate**: `F = 86` frames/sec  
- **Code “grid” shape** for an audio segment of duration `S` seconds:  
  - `T = ceil(F * S)` frames  
  - Discrete codes: `codes ∈ [0..1023]^(Nq × T)` (often called a “codegram”) citeturn2view0turn9view4  

Derived token throughput (useful for sizing models/inference time): **774 tokens/sec** (9×86), which is dramatically shorter than raw waveform modeling, and short enough that you can use transformer blocks without huge sequence lengths for one‑shots. citeturn2view0turn9view4

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Descript Audio Codec RVQGAN architecture diagram","EnCodec neural audio codec architecture diagram","SoundStream neural audio codec architecture figure"],"num_per_query":1}

### Freeze vs fine‑tune the codec (and what “fine‑tune” really buys you)

For v1, the safest recommendation is: **freeze DAC entirely** and train only the variation model on top of its latent representation. This aligns with how DAC is commonly used as a tokenizer in token‑space generation/restoration systems: e.g., MaskSR explicitly uses a **frozen** pretrained DAC to tokenize 44.1 kHz audio to a **9×T codegram** before a separate model predicts acoustic tokens. citeturn9view4turn1view4

If later you decide to fine‑tune the codec, do it only as a targeted optimization once you have evidence DAC reconstruction is the bottleneck for your source material. Two reasons to be cautious:

- Codec representations can be **fragile under repeated encode/decode** (“idempotence” issues): some neural codecs degrade noticeably after multiple encodings, motivating specialized fine‑tuning objectives to increase stability. citeturn10view1  
- Domain adaptation can cause **catastrophic forgetting** in neural networks, which is a well‑studied risk in sequential/domain fine‑tuning; mitigation often involves regularization or replay/mixing original data. citeturn20search8  

If you do fine‑tune, a conservative approach is: **adapters/LoRA‑style small parameter deltas + replay** (mix a large “general audio” sample with your drum set) rather than full fine‑tuning, to reduce forgetting risk. citeturn20search8

### How reconstruction quality degrades off‑distribution (field recordings, found sounds)

There’s no single universal guarantee here; what the literature does show is that neural codecs differ materially in robustness and can introduce **non‑linear distortions** and **spectral response changes**, especially under noise/real‑world conditions, and that training data diversity and quantization strategy influence robustness. citeturn7view0

DAC’s authors explicitly position it as a “universal” codec across speech/environment/music, and highlight design choices aimed at improving artifacts and fast transient modeling (e.g., multi‑scale mel reconstruction and low‑hop reconstruction being critical for fast transients). citeturn2view0turn6view3  
That makes DAC a defensible first choice for “arbitrary found sounds,” but you should still empirically check reconstruction on your intended inputs before committing.

## Latent‑space geometry and what evidence says about “small perturbations”

The follow‑up brief correctly challenges the assumption “small latent movement ⇒ small perceptual movement.” For *discrete* codec tokens, you should assume the opposite unless proven.

### Evidence that discrete codec token spaces are not perceptually smooth

Two lines of evidence from recent codec‑token research:

- **Discrete Representation Inconsistency (DRI)**: identical audio segments can tokenize to *different* discrete token streams depending on contextual factors (e.g., whether the segment is encoded with surrounding context). The authors describe this as the representation becoming “fragile” and “sensitive,” with “drastic drifts” in the token sequence from minor signal changes. citeturn9view2  
- **Idempotence / code drift**: state‑of‑the‑art neural codecs can degrade after only a small number of re‑encodings; the study frames idempotence as a stability property of the encoded representation under repeated encoding/decoding and shows large differences across codecs. citeturn10view1  

Both point to a key engineering conclusion:

**Token index “distance” is not a meaningful proxy for perceptual distance.**  
A “one‑step” token change (e.g., 123→124) is not inherently smaller than 123→900 because codebook indices are categorical. The only safe notion of “smallness” is either:
- a *sparsity* notion (few token changes), or
- an *audio‑domain* notion (decoded audio stays within a perceptual distance threshold).

### Mitigation strategy: design the variation model to learn perceptual constraints, not assume them

A v1‑appropriate mitigation is to make **audio‑domain perceptual distance the guardrail**, and treat latent/token edits as “free” as long as they remain within empirically derived bounds (see next section). This aligns with how recent discrete‑token generation systems constrain outputs: they do not rely on Euclidean smoothness of codes; they use masked‑token prediction plus iterative schedules, with sampling temperature and “confirm” dynamics to control variability. citeturn25view0turn26search4

### Should you add perceptual regularization that explicitly rejects “too large” perturbations?

Yes—*but do it in a way that is implementation‑tractable with discrete tokens.*

Rather than trying to penalize latent deltas directly (which is ill‑defined for categorical tokens), use one of these practical patterns:

- **Hard accept/reject sampling (simple, very effective for v1):**  
  1) sample K candidate variations in token space;  
  2) decode with DAC;  
  3) compute a perceptual distance to input (multi‑resolution STFT distance and/or MFCC distance);  
  4) accept those within your target band; repeat if needed.  
  This is a direct way to enforce “micro‑variation only,” with no differentiability requirements.

- **Soft regularization during training (optional v1.5):**  
  If you want waveform‑space losses while predicting tokens, you’ll need a differentiable relaxation (Gumbel‑Softmax or similar) rather than pure argmax sampling. EnCodec explicitly describes a **Gumbel‑Softmax quantizer** as a differentiable approximation for discrete selection, which is conceptually relevant if you later decide to backpropagate through decoding. citeturn5view0

## Implementable latent‑space variation model design

This section answers the “critical priority” implementation gap: what to build, concretely.

### Recommendation for v1: micro‑inpainting on DAC tokens (masked token modeling with a smallness prior)

The strongest “starting point” prior art for your exact direction is **TRIA (2025)**: a masked‑token transformer that generates drum audio by predicting masked DAC tokens, decoding back via DAC, and using a **SoundStorm‑style coarse‑to‑fine unmasking schedule**. TRIA also explicitly leverages the RVQ property that each codebook corrects residual error from earlier codebooks, and it forms a time sequence by **summing codebook embeddings** before passing them to a transformer. citeturn25view0

You do *not* need TRIA’s rhythm conditioning for round‑robin generation; but its tokenization + transformer + masked decoding mechanics are directly reusable.

#### Core architectural idea

Model `p(z_var | z_in)` where:
- `z_in` is the DAC codegram for the input one‑shot.
- `z_var` is a codegram for a *variation*.

Instead of generating a whole new codegram (high risk of identity drift), do **micro‑inpainting**:
- Copy most tokens from `z_in`.
- Only resample a small subset of tokens—preferably in **later RVQ codebooks** (fine detail), and optionally in tail frames more than attack frames.

This directly operationalizes your “smallness prior” as **sparsity of edits**, not “small numeric deltas.”

### Why “edit only later codebooks” is especially promising

RVQ is explicitly defined as sequentially quantizing the residual after prior quantization steps. citeturn4view0turn1view3turn25view0  
So later codebooks are naturally where “micro‑details” live. Empirically, DRI work also reports that inconsistency becomes more pronounced in deeper codebooks, consistent with the idea that later codebooks are more sensitive/fine‑grained. citeturn9view2

### Concrete delta model specification (inputs/outputs/shapes)

Assume:
- input audio length is padded/truncated to `S_max` seconds (recommend: **1.0 s** for v1; enough for most drum tails).
- DAC 44.1k: `Nq=9`, `F=86`, thus `T_max=86`.

**Inputs**
- `z_in`: int tensor `[B, Nq, T_max]` (0..1023)
- `mask`: boolean tensor `[B, Nq_edit, T_max]` where `Nq_edit ∈ {1..4}` (e.g., edit only codebooks 6–8 or 5–8)
- optional `noise_seed`: used only to generate random masks and/or sampling noise

**Model**
- embedding tables `E_i: [1024] → R^D` for each codebook `i` (TRIA does this) citeturn25view0  
- time‑step representation: `h_t = Σ_i E_i(z_in[i,t]) + E_mask(masked)` (sum embeddings across codebooks; TRIA does this) citeturn25view0  
- a small bidirectional transformer encoder over `t=1..T_max`
- per editable codebook `i`, a projection head to logits: `[B, T_max, 1024]`

**Outputs**
- logits `L_i[t]` for each editable codebook
- sampled tokens for masked positions (by categorical sampling with temperature)

### Smallness prior: three complementary, concrete mechanisms

You want “subtle but real.” In practice, you will need *both* a prior and a calibration target.

1. **Structural smallness: restrict where edits are allowed**  
   - Codebook restriction: only edit the last `Nq_edit` codebooks.  
   - Time restriction: define regions (attack window vs sustain/tail) and use lower mask probability on the first 5–15 ms equivalent frames.

2. **Statistical smallness: penalize edit rate**  
   Define:
   - `edit_rate = (# masked positions) / (Nq_edit * T_max)`  
   - `change_rate = mean( z_out != z_in )` on editable codebooks  
   Then add a penalty:
   - `L_small = λ * |change_rate - target_change_rate|`  
   where `target_change_rate` comes from real round‑robin pairs (see next section).

3. **Perceptual smallness: reject samples whose decoded audio is “too far”**  
   Decode with DAC and compute a perceptual distance (multi‑resolution spectral loss is standard in codec work; DAC also emphasizes it for modeling transients). citeturn2view0turn4view0turn6view3  
   Use this either as:
   - a hard rejection filter (v1), or
   - a soft constraint (v1.5 with differentiable relaxation)

### Stochasticity: what noise to use, and how to calibrate its magnitude

For masked token modeling, you get stochasticity from two places:

- **Random masking pattern** (which tokens are eligible to change)
- **Sampling noise** when choosing replacement tokens from logits (temperature/top‑p)

This is exactly how masked token generation methods like SoundStorm‑style inference and VampNet‑style nondeterministic unmasking operate at a high level (TRIA explicitly cites adopting those strategies). citeturn25view0turn26search4turn24search21turn26search4

A practical calibration method:
1. Precompute a distribution over **real round‑robin variation magnitude** (audio‑domain and token‑domain) from your libraries.
2. Tune:
   - `mask_prob`
   - which codebooks are editable
   - sampling temperature
   until generated variations match the *median and IQR* of that real distribution.

### Pseudocode for v1 micro‑inpainting generator

```text
# Constants for DAC 44.1k fullband
NQ = 9
CODEBOOK_SIZE = 1024
FPS = 86
T_MAX = 86          # 1.0s
EDIT_CODEBOOKS = [6,7,8]   # example: only last 3 codebooks
ATTACK_FRAMES = 2          # ~23ms at 86 Hz (tune)

# Inputs: wav_in (44.1k), seed
z_in = dac.encode(wav_in)                      # shape [NQ, T]
z_in = pad_or_trim(z_in, T_MAX)

# Build stochastic edit mask
mask = zeros([len(EDIT_CODEBOOKS), T_MAX])
for cb_idx in EDIT_CODEBOOKS:
  for t in range(T_MAX):
     p = p_attack if t < ATTACK_FRAMES else p_tail
     mask[cb_idx, t] = Bernoulli(p, seed++)

# Model forward pass
logits = variation_model(z_in, mask, seed)     # per editable cb: [T_MAX, 1024]

# Sample new tokens only where masked
z_out = z_in.clone()
for cb in EDIT_CODEBOOKS:
  for t in masked_positions(cb):
     z_out[cb, t] = categorical_sample(logits[cb, t], temperature, seed++)

wav_out = dac.decode(z_out)

# Optional acceptance filter (recommended for v1):
if perceptual_distance(wav_out, wav_in) not in target_band:
    repeat sampling (up to K tries) or lower temperature
return wav_out
```

The key engineering advantage: this avoids assuming any “Euclidean smoothness” in token space while still giving you a strong dial (mask rate / codebook choice) over micro‑variation.

## Training objective and whether you need waveform losses

### How to structure training pairs without leaking velocity layers

The v1 goal is micro‑variation at constant perceived intensity/timbre identity. For training on commercial libraries, keep training pairs within:
- same instrument articulation
- same velocity layer (as your initial brief required), so the model doesn’t conflate variation with dynamics.

(You can enforce this simply by forming RR pairs only within a velocity bin and excluding cross‑velocity pairs.)

### Supervision signal choices (token loss vs decoded audio loss)

The variation model “lives” in token space, so the most implementable v1 objective is **masked token cross‑entropy**:

- Sample random pairs `(A, B)` from a round‑robin set.  
- Encode both with DAC to `z_A`, `z_B`.  
- Train the model to predict **masked subsets of `z_B`** conditioned on `z_A` (and optionally the unmasked portion of `z_B`). This mirrors how masked token restoration models are trained. citeturn25view0turn9view4  

Why predict `z_B` rather than “make up something”? Because across many pairings, maximum likelihood training approximates the conditional distribution of plausible RR variations.

#### Do you need to backprop through the DAC decoder?

Not for token cross‑entropy training. If your training loss is purely categorical CE over the target tokens, the codec stays frozen and you never need differentiability through decoding.

If you want to *add* decoded waveform losses (to reduce artifacts), reality is:

- With **hard sampled tokens**, decoding is differentiable but sampling is not.
- To backprop end‑to‑end, you need either:
  - a straight‑through estimator, or
  - a differentiable categorical relaxation (e.g., Gumbel‑Softmax); EnCodec explicitly describes Gumbel‑Softmax quantization as a differentiable approximation technique in its appendix. citeturn5view0  

Given “musician‑developer, not MLOps engineer,” the recommended path is:

1) **v1**: token CE training + post‑hoc acceptance filtering using perceptual metrics.  
2) **v1.5** (only if needed): add differentiable relaxation and a multi‑resolution spectral loss on decoded audio (both EnCodec and DAC training emphasize multi‑scale spectral/perceptual losses for audio quality). citeturn4view0turn2view0  

### Should you include identity (A→A) pairs?

If you implement micro‑inpainting and constrain edits, you already have a strong similarity anchor. Adding identity pairs can help prevent drift, but it carries a real risk of encouraging copying.

A compromise pattern is: include identity pairs rarely (e.g., 5–10%) and train them with a **higher “edit penalty”** (force the model to learn that sometimes “no change” is valid), while still relying on stochastic sampling at inference to generate diversity.

### Curriculum strategy (useful and simple)

Start with easy constraints, then relax:

- Phase 1: edit only last 1–2 codebooks; low mask probability; high acceptance threshold (very small perceptual distance allowed).
- Phase 2: edit last 3–4 codebooks; slightly higher mask probability; acceptance band widened to match real RR baseline.

Curriculum learning as a general strategy is a well‑established optimization aid, even if audio‑specific curricula are task-dependent. citeturn20search21

## Data engineering specifics: transient alignment and splitting protocol

### Recommended transient alignment algorithm (sample‑accurate, automated)

A robust, implementable alignment pipeline for drum one‑shots:

1. **Coarse onset estimate via spectral flux onset strength**  
   Use an onset strength envelope based on spectral flux and peak picking (librosa provides exactly this: `onset_strength` and `onset_detect`). citeturn11search0turn11search27  

2. **Backtrack to a consistent “pre‑onset” reference**  
   Librosa provides `onset_backtrack`, which backtracks detected peaks to a preceding local minimum of an energy function—useful when you want to keep consistent pre‑transient noise. citeturn11search0  

3. **Fine alignment via short‑window cross‑correlation on the attack band**  
   After you have a coarse onset, take a short window around the attack (e.g., 5–20 ms), optionally high‑pass (to emphasize attack edge), then compute cross‑correlation with a reference hit’s attack window to estimate sample offset and shift accordingly.

This hybrid approach avoids the brittleness of “align to absolute peak” (which can differ across articulations) while still landing sample‑accurate alignment.

**Align to onset point vs peak?**  
For micro‑variation learning, aligning to the **onset point** (or a backtracked onset) is usually better than aligning to the absolute peak because it standardizes “when the event begins,” not “when the maximum happens,” which varies with head/brush/noise content. The `onset_backtrack` concept supports precisely that kind of consistent reference point. citeturn11search0  

**Library pointers**
- Librosa onset detection docs: practical, well‑documented, and directly implements spectral‑flux onset strength. citeturn11search0turn11search16  
- Madmom is a MIR‑focused Python library that includes onset detection pipelines and was explicitly designed for rapid audio workflow prototyping. citeturn11search28  

### Data splitting protocol that actually prevents leakage with rotation augmentation

The main leakage risk you flagged is real: if you form training pairs by rotating RR hits within a set, then naive random splitting will almost certainly put “siblings” from the same recording session into both train and test.

A concrete protocol:

- Define a **group key** at the highest granularity you can reliably extract:
  - `(library_id, kit_id, instrument_id, articulation_id, mic_perspective_id)`  
- Split *by group key*, not by individual wav.
- Ensure **no group key** appears in more than one split.

For evaluation intended to measure generalization beyond one recording context:

- **Dev split**: same libraries, different kits/instruments (fast iteration, stable).  
- **Test split**: hold out **entire libraries** (maximally different rooms/mics/preamps).  

This explicitly answers your question “should v1 generalize across recording conditions?”:
- If your immediate go/no‑go is “does this remove machine‑gun effect at all,” start with dev.  
- If your go/no‑go is “does this generalize to arbitrary one‑shots,” you need cross‑library test early—because codec/token behaviors and your variation model may implicitly learn studio‑specific artifacts.

### Evaluating on truly novel sounds (found sounds) without training data

Because you won’t have RR ground truth for field recordings, treat this as a separate “OOD evaluation pack”:

- Build a curated set of ~50–200 one‑shots (found metal hits, paper, plastic, foley), none of which appear in training.
- Evaluate:
  - similarity‑to‑input metrics (must stay high)
  - variation magnitude vs your RR baseline band (must be comparable)
  - manual listening spot checks

The robustness literature on neural codecs supports the idea that behavior can change in noisy/unseen scenarios, so explicitly measuring OOD behavior is necessary rather than assumed. citeturn7view0

## Automated iteration loop spec with LLM integration and guardrails

### Orchestration recommendation: Python experiment runner first, then optional sweeps

To balance simplicity and automation:

- Start with a **single Python “runner”** that:
  - reads a YAML/JSON config,
  - launches training,
  - runs a fixed eval suite,
  - writes a metrics JSON artifact per run.

Then layer on:
- **Weights & Biases Sweeps** for hyperparameter search when you’re ready (it’s designed specifically for automated hyperparameter search and experiment tracking). entity["company","Weights & Biases","ml experiment tracking"] citeturn12search0  
- Or **Optuna** if you prefer local, code‑first HPO (Optuna’s study/trial API is designed for programmable search). citeturn12search1turn12search16  

Using full Vertex AI Pipelines is likely overkill for v1 unless you already have MLOps muscle memory; it’s explicitly positioned as an orchestration layer for ML workflows and adds conceptual overhead. citeturn12search2  

### How to invoke Claude Code in the loop (practical, programmatic)

Your brief calls out Claude Code as the engineering collaborator. The most direct way to integrate it is via the **Claude Code “headless” Agent SDK**, which is explicitly intended to run Claude Code programmatically from CLI/Python/TypeScript. citeturn12search3turn12search22

### JSON metric report format (example)

A metric report that tends to work well with LLM diagnosis is:

- run metadata
- scalar summary metrics
- baseline bands (median/IQR from real RR)
- trend deltas vs previous runs
- “top failures” with links/filenames

Example (one run):

```json
{
  "run_id": "2026-02-28T14-22-10Z_r17",
  "git_commit": "abc1234",
  "codec": {"name": "DAC_44k_full", "Nq": 9, "fps": 86, "codebook_size": 1024},
  "model": {"type": "micro_inpaint_transformer", "editable_codebooks": [6,7,8], "D": 256, "layers": 6},
  "train": {"steps": 120000, "batch_size": 64, "lr": 2e-4},
  "inference": {"mask_p_attack": 0.02, "mask_p_tail": 0.08, "temperature": 0.9, "k_candidates": 8},
  "baselines": {
    "rr_token_change_rate": {"median": 0.11, "iqr": [0.07, 0.16]},
    "rr_mrstft_distance": {"median": 0.18, "iqr": [0.12, 0.25]}
  },
  "metrics": {
    "similarity": {
      "mrstft_to_input": {"mean": 0.16, "p95": 0.24},
      "mfcc_dist_to_input": {"mean": 12.3, "p95": 18.1}
    },
    "variation": {
      "token_change_rate": {"mean": 0.09},
      "pairwise_mrstft_between_vars": {"mean": 0.14}
    },
    "artifact_flags": {
      "attack_smear_score": {"mean": 0.07},
      "tonal_ringing_events": 3
    }
  },
  "regressions_vs_prev": [
    {"metric": "attack_smear_score.mean", "delta": "+0.03", "severity": "high"}
  ],
  "audio_examples": [
    {"id": "snare_013", "paths": {"in": "examples/snare_013_in.wav", "vars": ["...v1.wav","...v2.wav"]}}
  ]
}
```

### Example of the LLM’s “allowed” output (guardrailed)

Have the LLM return:

- “hypothesis”
- “1–3 changes”
- “why these changes”
- “rollback criteria”

Example:

```json
{
  "diagnosis": "Variation is low vs RR baseline; attack smear increased, likely from masking too early in the onset frames.",
  "proposed_changes": [
    {"type": "hp_tune", "param": "mask_p_attack", "from": 0.02, "to": 0.005},
    {"type": "hp_tune", "param": "editable_codebooks", "from": [6,7,8], "to": [7,8]},
    {"type": "hp_tune", "param": "temperature", "from": 0.9, "to": 0.8}
  ],
  "rollback_if": [
    "mrstft_to_input.p95 > 0.28",
    "attack_smear_score.mean > 0.08"
  ]
}
```

### Guardrails to prevent destructive changes

Given Claude Code can edit and run commands, keep it boxed in:

- **Phase gating**: first 20–50 runs allow only hyperparameter edits; architecture edits require human sign‑off.
- **Auto‑rollback**: if 3–5 key metrics regress beyond thresholds, revert to last “good” git commit.
- **Diff budget**: cap file edit size or require tests to pass before running training.
- **Max unattended iterations**: hard stop after N runs to force a listening check.

Also note that security researchers have recently highlighted vulnerabilities in Claude Code workflows; regardless of whether those specific issues apply to you, it’s a strong reason to keep the loop sandboxed and version‑controlled. citeturn12news37  

## Post‑processing pipeline for 44.1 kHz / 16‑bit / stereo WAV delivery

Even when using a 44.1 kHz codec, you should treat decoder output as **float** and apply a deterministic finishing chain.

A recommended post chain:

1. **DC offset removal**  
   - simplest: subtract mean per channel (safe for one‑shots)  
   - or a very low‑cut high‑pass (e.g., 10–20 Hz) if needed  
   DC offsets are a known practical issue in audio pipelines and are commonly removed because they can reduce headroom and cause downstream problems. citeturn13search3  

2. **Level strategy: preserve input level by default**  
   For round‑robin variation, you usually want *some* natural level differences, but you don’t want the model to “cheat” by varying only gain.  
   Suggested default: match **RMS over a fixed window after onset** (e.g., 10–200 ms), with a clamp (±1 dB), and don’t peak‑normalize each variation independently.

3. **Tail safety fades** (only if you hard‑truncate)  
   If the pipeline trims/pads, apply a short equal‑power fade‑out (5–20 ms) only when truncation happens to avoid clicks.

4. **Resampling** (only if needed)  
   If any part of the pipeline produces 48 kHz (e.g., EnCodec route), resample with a high‑quality band‑limited sinc method. Librosa’s resampler explicitly notes using **soxr_hq** by default, and Python‑SoXR is a wrapper to libsoxr for high quality sample rate conversion. citeturn13search16turn13search6  

5. **Dither to 16‑bit PCM at final export**  
   When reducing bit depth, apply **TPDF dither** at the end of the chain (after all DSP). Classic AES‑community work argues triangular‑PDF dither eliminates distortion and noise modulation artifacts in quantization. citeturn13search0turn13search1  

## Timeline, milestone gates, and cloud cost envelope

### Realistic v1 timeline (AI‑assisted development, assuming data access is ready)

These are “calendar time” estimates for a single focused developer with an LLM coding assistant:

- **Data prep and alignment pipeline**: 5–10 days  
  (build import, trimming, onset+cross‑corr alignment, grouping metadata, split strategy, caching codegrams)

- **First end‑to‑end training run with listenable outputs**: 3–7 days after data pipeline  
  (tokenize → train micro‑inpainting transformer with CE → decode → generate 4–8 vars per input)

- **Iteration to pass a machine‑gun proxy test**: 2–4 weeks  
  (you’ll spend most time tuning mask schedules, codebook edit strategy, and acceptance thresholds)

- **Cross‑library generalization pass + OOD found‑sound eval**: 1–2 weeks  
  (often reveals whether you learned “studio fingerprints”)

These ranges are consistent with the fact that similar discrete‑token drum generation work (TRIA) trains transformer token models with DAC decoding and describes explicit GPU training time per model; TRIA reports training on A10G GPUs and gives a concrete “hours per model” figure, which is useful as an order‑of‑magnitude anchor for how long token transformers can take even with compact representations. citeturn25view0  

### Go/no‑go gates (when to pivot)

Gate the project early on measurable outputs:

- **Gate A (after first listenable outputs):**  
  Generated variations must preserve transient integrity—no obvious ringing or smeared attack on at least snares/kicks. If you can’t get this after restricting edits to late codebooks and adding acceptance filtering, the approach may be fighting codec/manifold issues.

- **Gate B (after baseline calibration):**  
  Your generated variation magnitude should stay within the IQR band of real RR variation (audio‑domain metrics) while exceeding a minimum variation threshold (to avoid copies). If you can’t simultaneously satisfy both axes, pivot to a non‑ML approach (procedural micro‑jitter / convolutional perturbation) for v1 and revisit ML later.

- **Gate C (cross‑library):**  
  If cross‑library generalization collapses, you likely need either more diverse training data or additional conditioning/normalization (e.g., library‑style embeddings), which may be out‑of‑scope for a quick v1.

### Cost envelope (order‑of‑magnitude)

Official pricing pages for some Google Cloud ML components were returning server errors during this research session, so the numbers below use third‑party price aggregators that expose per‑hour pricing for specific machine types and explicitly show update timestamps. (You should still verify in your own billing console before spending.) citeturn17search5turn17search3turn17search12turn17search22

Two useful reference instance types:

- **L4 (G2) class**: `g2-standard-8` is reported around **$0.85–$1.36/hr** depending on region in one aggregator. citeturn17search5  
- **A100 40GB (A2) class**: `a2-highgpu-1g` is reported around **$3.67/hr** on‑demand in at least one published calculator snapshot, with region variance. citeturn17search12turn17search22  

A rough POC budget scenario:

- Data prep + token caching: 20–50 GPU‑hours is usually unnecessary; do it on CPU.  
- Model training + sweeps:  
  - 10 runs × 8 hours/run × $3.67/hr ≈ **$294** (A100 route), plus storage/egress. citeturn17search12turn17search22  
  - If you can run on L4 for early development, you can cut cost substantially at the expense of speed.

As a sanity check, note that EnCodec itself reports training its models using **8 A100 GPUs**, which underscores that codec training is expensive—but your variation model is far smaller than training a codec from scratch. citeturn19view2  

## Prior art most relevant to round‑robin variation

### Directly reusable prior art patterns

- **TRIA (2025): masked DAC‑token transformer for drums**  
  TRIA tokenizes audio with DAC, sums per‑codebook embeddings into a time sequence, and performs SoundStorm‑style iterative masked token prediction, decoding back via DAC. This is conceptually extremely close to what you need—except your conditioning is “input hit identity,” not rhythm prompts. citeturn25view0  

- **VampNet (2023): masked acoustic token modeling for variation**  
  VampNet is explicitly framed as masked token modeling for music “variation” (among other tasks) using a bidirectional transformer and multiple sampling passes. Even if you don’t reuse its pretrained weights, the prompting/sampling design is directly relevant. citeturn26search4turn26search0turn26search1  

- **ICLR 2026 discrete diffusion for token inpainting**  
  Token‑based audio inpainting via discrete diffusion is an emerging direction (ICLR 2026 poster) that applies diffusion over tokenized music representations and introduces regularization and structured corruption strategies for spans. While it targets restoration, the mechanism is compatible with “micro‑inpainting for variation.” citeturn27view0  

### The “snare performance variation as a timbre case study” referenced in your brief

The paper **“Real‑time Timbre Remapping with Differentiable DSP” (2024)** explicitly motivates work by noting repeated retriggering (“machine‑gun effect”), and uses **snare drum performance variation** as a case study for learning mappings that preserve and recreate subtle timbral differences. It references prior findings that drummers systematically vary intensity and timbre within grooves, and it builds a differentiable synthesizer inspired by the TR‑808 as a target system. citeturn23view0turn23view1  

This isn’t a codec‑latent round‑robin generator, but it strongly supports the premise that **subtle, structured micro‑variation is musically meaningful and learnable**—and it offers engineering ideas (feature difference losses; onset‑anchored processing) that translate well to your evaluation and alignment needs. citeturn23view2  

### Is there published ML explicitly for “round‑robin from one‑shot”?

I did not find a well‑cited, established academic line that directly claims “generate round‑robin sets from a single one‑shot” as its primary contribution. The closest high‑signal adjacent work is:
- masked token drum generation with DAC (TRIA) citeturn25view0  
- masked token modeling for variation (VampNet) citeturn26search4  
- token inpainting/diffusion for restoration (ICLR 2026) citeturn27view0  

That’s not a red flag: it just means you’re working in a space where **the prior art is adjacent rather than on‑the‑nose**, which increases the value of your baseline‑driven metric calibration and hard go/no‑go gates. citeturn7view0turn9view2turn10view1  

At the end of the day, the most implementation‑credible v1 stack is:

- **DAC 44.1k frozen tokenizer/decoder** citeturn2view0turn1view0  
- **micro‑inpainting masked token transformer** (TRIA/VampNet‑style mechanics, but constrained to late codebooks and low mask rates) citeturn25view0turn26search4  
- **audio‑domain acceptance filtering** to enforce “subtle only,” because token spaces are not reliably smooth citeturn9view2turn10view1  
- **automated report loop** driven by JSON artifacts and a headless Claude Code integration, with strict guardrails citeturn12search3turn12search0  

> Quote corner (for inspiration, not implementation): “Although audio with or without contextual audio is encoded into different discrete speech token sequences, both sequences can be used to reconstruct the original audio information…” citeturn9view2