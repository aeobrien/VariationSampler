**VARIATION SAMPLER**

Implementation Notes & Practical Guidance

Supplementary context for technical planning

February 2026

This document captures practical insights, recommendations, and caveats that emerged during the project scoping and research commissioning process. It supplements the two formal research reports and the vision statement. Much of this material is not covered in the reports but is important for anyone building the technical plan.


# **1. Architecture Decisions Already Made**

Two rounds of research have converged on a specific technical approach. These decisions should be treated as settled for v1 unless empirical results force a pivot.

- **Codec:** DAC (Descript Audio Codec) 44.1 kHz, full-bandwidth configuration. 9 RVQ codebooks, 1024 entries per codebook, 86 Hz frame rate. Frozen for v1 — do not fine-tune unless reconstruction quality on your specific source material is demonstrably the bottleneck.

- **Variation model:** Masked-token micro-inpainting transformer. Edits only the later codebooks (start with 6–8, the last three). Trained with masked token cross-entropy on round-robin pairs. The “smallness prior” is structural (sparse edits in late codebooks) rather than numeric (no assumption about Euclidean smoothness in token space).

- **Quality control:** Audio-domain acceptance filtering. Decode every candidate variation, compute perceptual distance to input, reject anything outside a calibrated band derived from real round-robin data. This is the primary guardrail against synthetic-sounding output.

- **Training framework:** PyTorch. The ecosystem support (Diffusers, AudioCraft, DAC repo) is overwhelmingly PyTorch-first.

- **Cloud:** Google Cloud. A100 (A2 series) for training, L4 (G2 series) or T4 for inference and prototyping. Vertex AI Model Registry for version tracking.


# **2. The Key Insight the Research Revealed**

The most important finding across both reports is that codec token spaces are not perceptually smooth. Small changes in token index do not reliably produce small changes in perceived audio, because codebook indices are categorical — entry 123 is not inherently “closer to” entry 124 than to entry 900.

This means the original intuition of “add a small delta in latent space” would likely not work. The micro-inpainting approach sidesteps this entirely by reframing the problem: instead of asking “what small change produces a subtle variation?”, it asks “what tokens would plausibly appear here if this were a different recording of the same hit?” The model learns to predict alternative tokens from the training distribution, and the acceptance filter enforces that the decoded result stays perceptually close.

This is the single most important architectural insight. Any implementation plan that reverts to “interpolate in latent space” or “add Gaussian noise to embeddings” is likely to fail or produce synthetic-sounding results.


# **3. Practical Considerations Not Fully Covered in the Reports**

## **3.1  Cache DAC encodings upfront**

This is easy to overlook but has significant impact on iteration speed. Encoding the full training dataset through DAC (audio → codegrams) should be done once as a preprocessing step, with the resulting token tensors saved to disk. Every subsequent training run should load cached codegrams directly, never re-encoding audio. DAC encoding is not instantaneous, and repeating it on every run wastes GPU time and adds unnecessary complexity to the training loop.


## **3.2  The late-codebook restriction may be too conservative**

The reports recommend editing only codebooks 6–8 (the last three of nine). This is a sensible starting point because later codebooks encode finer residual detail. However, real round-robin variation isn’t exclusively fine detail — it includes subtle timing micro-shifts, slight phase differences from different strike angles, and gross spectral envelope changes that may be encoded in earlier codebooks.

If late-codebook-only editing produces variations that sound like the same recording with slightly different noise texture (rather than genuinely different hits), the solution is to cautiously open up earlier codebooks — perhaps 4–8, or even 3–8 — while keeping the mask probability very low for earlier codebooks and relying on acceptance filtering to reject anything too different. The go/no-go gates should catch this, but the implementation plan should anticipate it as a likely tuning axis.


## **3.3  Acceptance filtering rejection rate**

The acceptance filtering loop (generate K candidates, decode, check perceptual distance, keep or reject) is the primary quality control mechanism. Neither report estimates what a reasonable acceptance rate target is, or what to do if it’s very low.

If acceptance rate is below \~20%, inference becomes expensive and slow (you’re generating 5x more candidates than you keep). This suggests either the model is poorly calibrated (mask rate or temperature too high) or the acceptance band is too narrow. The iteration loop should track acceptance rate as a first-class metric and treat sustained low rates as a signal to adjust generation parameters, not just a cost to absorb.


## **3.4  Goodhart’s Law: what it actually looks like here**

Both reports warn about the risk of optimising metrics at the expense of perceptual quality, but don’t give concrete examples. Here’s what to watch for:

- A model that adds subtle broadband noise to each variation — technically different by spectral metrics, within the acceptable distance band, but sounds like tape hiss rather than a different drum hit.

- A model that shifts the timing of the tail/decay by a few samples — metrically distinct, perceptually identical or worse (introduces phase cancellation issues when layering).

- A model that alters only inaudible high-frequency content above 15 kHz — spectrally different, completely imperceptible.

The best defence is the periodic listening check-ins. No metric suite can fully substitute for a trained ear evaluating whether the output sounds like “a different hit” vs. “the same hit with artifacts.”


## **3.5  The automation loop should be simpler than the reports suggest**

The reports describe a relatively complex LLM-in-the-loop system with Claude Code headless mode applying code changes. For v1 of the automation, this introduces unnecessary risk and complexity. The recommended approach is significantly simpler:

- The automated loop is a Python script running on a GPU VM.

- Each iteration: train (or continue training) → generate test samples from held-out inputs → compute all metrics → write a structured JSON report.

- The JSON report is sent to Claude via the API (a standard messages API call, not Claude Code).

- **Claude’s role is constrained to returning a new config JSON** containing only hyperparameter adjustments: mask probability, temperature, editable codebook range, learning rate, acceptance thresholds. No code edits. No architecture changes. Just config.

- The runner script applies the new config and starts the next iteration.

- Guardrails: auto-rollback if key metrics regress past thresholds; hard stop after N iterations without improvement; hard stop after M total unattended iterations.

This is dramatically simpler to build, debug, and trust than a code-editing agent. Architecture changes and code modifications should be reserved for human-supervised sessions, triggered by the periodic listening check-ins.

Code-level changes via Claude Code (with the project owner supervising) happen between automated run batches, not within them.


## **3.6  Listening check-in cadence**

A practical rhythm: let the automated loop run 5–10 iterations unattended, then spend 20–30 minutes listening to the best and worst outputs from that batch. Focus on:

- Does the best output sound like a different hit, or like a processed version of the same hit?

- Does the worst output reveal a systematic failure mode (e.g., transient smearing, tonal artifacts)?

- Are the variations “interesting” in the way real round-robins are, or do they feel mechanical/algorithmic?

After listening, write a brief natural-language note summarising what you heard. This gets appended to the next Claude API prompt alongside the metrics JSON, so the LLM has both quantitative and qualitative signal. This is the Goodhart’s Law mitigation — your ears overrule the metrics.


## **3.7  Weights & Biases is worth setting up from day one**

W\&B gives you training curves, metric comparisons across runs, and audio sample logging essentially for free. When you do your periodic listening check-ins, having a dashboard that shows which runs improved on which metrics (and lets you play audio samples inline) makes the process dramatically more efficient than digging through directories of WAV files. It also provides the experiment history that makes the Claude API calls more useful — you can include trend data across many runs rather than just the last one.


# **4. Data Preparation Notes**

## **4.1  Training data scale**

The reports estimate minimum viable at \~500–1,500 round-robin sets for the proof of concept (starting with one instrument family, e.g., snares). With N=5 round-robins per set, that’s 10,000–30,000 training pairs via the rotation strategy. Recommended for cross-drum-family generalisation is 5,000–15,000 sets. The project owner has access to extensive commercial deeply-sampled libraries, so data availability should not be a bottleneck.


## **4.2  Data preparation pipeline**

The pipeline, in order:

- Import WAV files from commercial libraries, organised by library/kit/instrument/articulation/velocity bin.

- Trim silence with conservative margins (never clip pre-transient noise).

- Transient-align using spectral flux onset detection (librosa), backtracking to pre-onset reference, then fine alignment via short-window cross-correlation on the attack band.

- Pad or truncate to fixed length (1.0 second recommended for v1).

- Peak or loudness normalise within a narrow tolerance (±1 dB) to reduce level as a confound without destroying natural micro-dynamics.

- Encode through DAC and cache the resulting codegrams as tensors on disk.

- Compute and cache ground-truth baseline metrics: within-set pairwise distances (MR-STFT, MFCC/MCD) for every round-robin set, aggregated into distributions by instrument family.


## **4.3  Splitting protocol**

Split by group key (library\_id, kit\_id, instrument\_id, articulation\_id, mic\_perspective\_id). No group key may appear in more than one split. Two evaluation splits:

- **Dev split:** Same libraries, different kits/instruments. For fast iteration.

- **Test split:** Hold out entire libraries. For testing generalisation across recording conditions.

Additionally, build an out-of-distribution evaluation pack: \~50–200 one-shot found sounds (metal hits, paper, plastic, foley) with no round-robin ground truth. Evaluate similarity-to-input (must stay high) and variation magnitude vs. real round-robin baseline (must be comparable).


## **4.4  Velocity layers: exclude from v1**

Keep training pairs strictly within the same velocity bin. Do not include cross-velocity pairs. The model should learn micro-variation at constant perceived intensity, not dynamics. Velocity layer generation is a different problem for different research.


# **5. Go/No-Go Gates**

The project is designed to fail fast. Three gates, in order:

**Gate A: First listenable outputs (target: 2–3 weeks from start).** Generated variations must preserve transient integrity — no obvious ringing, smeared attacks, or tonal artifacts on snares and kicks. If restricting edits to late codebooks and applying acceptance filtering cannot produce clean transients, the approach may be fundamentally fighting codec or manifold issues. Decision: continue, adjust codebook range, or pivot.

**Gate B: Baseline calibration (target: 4–6 weeks).** Generated variation magnitude must fall within the IQR of real round-robin variation (measured by MR-STFT and MFCC distance), while simultaneously exceeding a minimum variation threshold (no copies). If both axes cannot be satisfied simultaneously, pivot to a non-ML approach (procedural micro-jitter / convolutional perturbation) for v1. This is the core feasibility gate.

**Gate C: Cross-library generalisation (target: 6–8 weeks).** Test on held-out libraries with different recording conditions. If generalisation collapses, the model has learned studio fingerprints rather than variation patterns. Decision: add more diverse training data, add conditioning/normalisation, or accept library-specific models as a valid v1 scope reduction.


# **6. Post-Processing Chain**

Applied to codec decoder output before delivery as final WAV:

- DC offset removal (subtract mean per channel).

- Level matching: match RMS over a fixed window after onset (10–200 ms) to the input sample, with a clamp of ±1 dB. Do not peak-normalise independently per variation (this would destroy natural level variation).

- Tail fade-out: apply a short equal-power fade (5–20 ms) only when hard truncation occurred.

- Dither to 16-bit PCM using TPDF dither, applied as the final step after all other processing.

- Export as 44.1 kHz / 16-bit / stereo WAV.


# **7. Estimated Cloud Costs**

Order-of-magnitude estimates for the full proof-of-concept phase:

- Data preparation and DAC encoding: mostly CPU work, minimal GPU cost.

- Training and sweeps: approximately 10 runs × 8 hours/run on A100 at \~$3.67/hr = \~$300. Early development on L4 or T4 can reduce this at the expense of iteration speed.

- Inference/generation for test samples: minimal incremental cost if run on the same VM.

- Total POC budget: roughly $300–$600 in compute, plus storage and egress.

This does not include Claude API costs for the automated loop, which depend on prompt length and iteration count but should be modest relative to GPU costs.


# **8. Key Prior Art References**

The most directly relevant prior art, ranked by implementation value:

- **TRIA (2025):** Masked-token transformer for drum audio generation using DAC codegrams. The closest existing system to what Variation Sampler needs. Its tokenisation, embedding (sum codebook embeddings per frame), and SoundStorm-style coarse-to-fine unmasking mechanics are directly reusable. Differs in that TRIA uses rhythm conditioning rather than input-hit conditioning.

- **VampNet (2023):** Masked acoustic token modeling for music variation using a bidirectional transformer. The prompting and sampling design (multiple sampling passes, temperature control) is directly relevant to controlling variation magnitude.

- **DAC (Descript Audio Codec):** The chosen codec. 44.1 kHz native, stereo support, open-source with pretrained weights. Universal design across speech/environment/music.

- **Real-time Timbre Remapping with DDSP (2024):** Uses snare drum performance variation as a case study. Supports the premise that subtle micro-variation is musically meaningful and learnable. Offers engineering ideas for onset-anchored processing and feature difference losses.

No published work was found that directly claims “generate round-robin sets from a single one-shot” as its primary contribution. The prior art is adjacent, not on-the-nose. This increases the importance of the empirical go/no-go gates.


# **9. A Note on Velocity Layer Generation (v2)**

Velocity layer generation is explicitly deferred, but it’s worth recording why it’s a fundamentally different problem so this context isn’t lost.

Round-robin variation is about micro-perturbation: the same physical event with tiny incidental differences. Velocity layer generation requires the model to simulate what happens when a membrane or surface is struck with different force — the harmonic content changes, the transient shape changes, the decay profile changes. That’s not variation, it’s synthesis. The training signal is completely different, the evaluation criteria are different, and it’s not clear that the same architecture (or even the same approach) would apply.

If v1 succeeds, velocity generation becomes a natural next research question. But it should be treated as a separate project with its own research phase, not as a feature addition to v1.


# **10. Overall Confidence Assessment**

From the research reports, summarised honestly:

- **Confidence that the model can generate “different” outputs:** _High._ Multiple sampling seeds in a generative model almost guarantees diversity.

- **Confidence that outputs remain “same source” while being “different enough”:** _Medium._ This is the hard part. The micro-inpainting approach with acceptance filtering is the most credible path, but it’s untested for this specific application.

- **Confidence that results will consistently beat hand-designed humanisation/randomisation:** _Medium-low until tested._ Artifacts are easy to introduce at the transient, and “subtle but natural” is a high bar.

The project is designed around this uncertainty. The gates are early, the scope is narrow, and the failure criteria are concrete. This is a genuine experiment, not an implementation of a proven technique.
