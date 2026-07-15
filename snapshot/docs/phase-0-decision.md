# Phase 0 Decision: Build vs Adapt

## Summary

Two open-source systems are strong candidates for adaptation. Both have MIT-licensed code (weights are CC BY-NC-SA 4.0, but we'd train our own). After evaluating both, the recommendation is to **adapt TRIA as the primary foundation**, with specific techniques borrowed from VampNet.

---

## Candidate Evaluation

### TRIA (The Rhythm In Anything)

**Repo:** [github.com/interactiveaudiolab/tria](https://github.com/interactiveaudiolab/tria)
**Paper:** O'Reilly et al., ISMIR 2025. "Audio-Prompted Drums Generation with Masked Language Modeling"
**Code licence:** MIT | **Weights licence:** CC BY-NC-SA 4.0
**From:** Interactive Audio Lab, Northwestern University

| Property | Value | Match to Our Spec |
|---|---|---|
| Codec | DAC 44.1 kHz | Exact match |
| Codebooks | 9 RVQ, 1024 entries each | Exact match |
| Frame rate | 86 Hz | Exact match |
| Architecture | 12-layer bidirectional transformer, D=512, 8 heads, 43M params | Close — we'd start smaller (D=256, 4–6 layers) but can scale |
| Embedding | Sum codebook embeddings per frame | Exact match to our spec |
| Decoding | SoundStorm-style coarse-to-fine iterative unmasking | Directly reusable |
| Training data | Drum stems (MusDB18-HQ + optional expansions) | Domain match |
| Training cost | 100k iterations, ~27 hours on 4x A10G | Within our budget |

**What we'd reuse:**
- DAC integration (encoding, decoding, codegram handling) — already working at our exact spec
- Transformer architecture (embedding tables, sum-per-frame strategy, projection heads)
- SoundStorm-style iterative unmasking schedule
- Training infrastructure (argbind config system, training loop patterns)
- Data augmentation pipeline

**What we'd change:**
- Remove rhythm conditioning (TRIA generates drums to match a rhythm prompt)
- Add input-hit conditioning (our model sees one hit and predicts variation tokens)
- Restrict masking to late codebooks (TRIA masks broadly; we mask sparsely in codebooks 6–8)
- Add separate mask probabilities for attack vs tail frames
- Add acceptance filtering pipeline (not in TRIA)
- Add our evaluation metric suite

**Estimated adaptation effort:** 1–2 weeks. The changes are well-scoped: conditioning mechanism swap + mask restriction + new evaluation. The core transformer and DAC plumbing stay intact.

---

### VampNet

**Repo:** [github.com/hugofloresgarcia/vampnet](https://github.com/hugofloresgarcia/vampnet)
**Paper:** Flores Garcia et al., ISMIR 2023. "Music Generation via Masked Acoustic Token Modeling"
**Code licence:** MIT | **Weights licence:** CC BY-NC-SA 4.0
**From:** Interactive Audio Lab, Northwestern University (same lab as TRIA)

| Property | Value | Match to Our Spec |
|---|---|---|
| Codec | DAC 44.1 kHz (custom config) | Partial — uses 14 codebooks, not 9 |
| Codebooks | 14 RVQ (4 coarse + 10 fine) | Does NOT match (we need 9) |
| Frame rate | 57 Hz | Does NOT match (we need 86) |
| Architecture | Two-stage: coarse (20 layers) + c2f (16 layers) | Heavier than we need |
| Training data | Music (not drums specifically) | Domain mismatch |

**Strengths:**
- Native inpainting support via `build_mask()` + `vamp()` API — exactly the UX pattern we want
- Proven adaptable to new domains (WhAM whale project, NeurIPS 2025)
- Confidence-based iterative decoding with Gumbel noise — good sampling technique
- Clean `Interface` class wrapping encode/mask/vamp/decode

**Weaknesses:**
- DAC configuration does NOT match our spec. Changing from 14 to 9 codebooks ripples through the entire model — different embedding tables, different coarse/fine splits, different iteration schedules. This is not a config change; it's an architectural change.
- Two-stage (coarse + c2f) architecture adds complexity we don't need for micro-inpainting where most tokens are copied from the input.
- Trained on music, not drums — we'd retrain anyway, but TRIA's drum-specific training data and augmentations are more relevant.

**What we'd borrow (as techniques, not code):**
- The `build_mask()` concept for flexible mask specification
- Confidence-based iterative decoding with temperature annealing
- The `periodic_prompt` and `upper_codebook_mask` masking patterns as inspiration for our attack/tail mask scheduling

**Estimated adaptation effort:** 2–4 weeks. Significantly more work because the DAC config mismatch requires reworking the model architecture, not just the conditioning.

---

### Other Systems Evaluated

| System | Verdict |
|---|---|
| **AudioCraft (Meta)** | EnCodec-based, autoregressive. Useful utility code but wrong codec and wrong generation paradigm. Not a starting point. |
| **SoundStorm (lucidrains)** | Clean reference implementation of MaskGIT over RVQ tokens. Useful for understanding the algorithm but no pretrained weights, no DAC integration. Study, don't adapt. |
| **AIDD (ICLR 2026)** | Discrete diffusion for token inpainting. Code not yet released. Uses WavTokenizer (single codebook), not DAC. Watch but not actionable. |
| **DAC repo** | The codec itself. MIT licensed, well-documented API. We use this regardless of which generation model we build on. |

---

## Recommendation: Adapt TRIA

**TRIA is the strongest starting point because:**

1. **DAC configuration is an exact match.** 9 codebooks, 1024 entries, 86 Hz, 44.1 kHz. This is the single most important factor — it means every tensor shape, every embedding table, every iteration schedule works out of the box. VampNet's 14-codebook setup would require rearchitecting.

2. **Already trained on drums.** TRIA's training pipeline, data augmentation (noise, bandpass, pitch shift, phase shift, EQ), and evaluation are all drum-specific. We're not adapting a music system to drums — we're adapting a drum system to a different drum task.

3. **Same lab, same lineage.** TRIA and VampNet are from the same research group (Interactive Audio Lab, Northwestern). TRIA explicitly builds on VampNet's masked token approach. TRIA is essentially "VampNet adapted for drums with standard DAC config" — it already did the adaptation work we'd otherwise do ourselves.

4. **Simpler architecture.** Single-stage 12-layer transformer vs VampNet's two-stage 36-layer system. For micro-inpainting where most tokens are copied, we want the lightest viable model.

5. **Well-scoped modifications needed.** The changes to go from "rhythm-conditioned drum generation" to "input-conditioned micro-inpainting" are clear:
   - Replace rhythm feature extraction with input codegram conditioning
   - Restrict masking to late codebooks with attack/tail scheduling
   - Add acceptance filtering
   - Add our metric suite

**From VampNet, adopt these techniques (implement in TRIA's framework):**
- Flexible mask building (attack vs tail probabilities, codebook-level control)
- Confidence-based iterative decoding with temperature annealing
- The interface pattern: clean encode → mask → generate → filter → decode pipeline

**From lucidrains/SoundStorm, study:**
- The level-by-level masking strategy (mask one RVQ level + all deeper levels) as an alternative to TRIA's approach, if we need more control over coarse-to-fine generation

---

## Knowledge Gaps Identified

The research reports and TRIA/VampNet codebases together cover most implementation needs. Remaining gaps:

1. **DAC stereo handling.** Our spec requires stereo output. TRIA appears to be mono. DAC supports stereo but we need to verify: does the 9-codebook config apply per-channel or jointly? How does stereo affect codegram shape? This needs a quick empirical test in Phase 1.

2. **Acceptance filtering calibration.** Neither TRIA nor VampNet implements the rejection-sampling acceptance filter that is central to our quality control. We need to build this from scratch, calibrated against real RR baseline distributions. This is Phase 3 work but should be kept in mind during Phase 2.

3. **Input-hit conditioning mechanism.** TRIA conditions on rhythm features extracted from a prompt. We condition on the input hit's codegram directly. The simplest approach: the input codegram IS the context (unmasked tokens), and the model predicts replacements for masked tokens. This is actually simpler than TRIA's conditioning — it's standard masked language modeling where the "prompt" is the input tokens themselves. Need to verify this works in practice.

---

## Next Steps

1. **Clone TRIA repo** and set up locally
2. **Run TRIA's existing pipeline** end-to-end to verify everything works: encode drum audio → train (even briefly) → generate → decode
3. **Study the code** in detail: understand the training loop, masking schedule, inference pipeline
4. **Begin modifications** in Phase 2: strip rhythm conditioning, implement input-hit conditioning, restrict masking
