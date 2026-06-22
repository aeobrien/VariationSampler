# VariationSampler — Ledger

> ML system that generates perceptually convincing round-robin variations from a single one-shot drum sample. Given one recording of a drum hit, produces 4-8 variations that sound like the same instrument struck again — not replayed, not processed. V1 is a proof of concept answering whether this is feasible.

## Status

**Phase:** Pre-development — vision and research complete, implementation not yet started
**Last updated:** 2026-04-07

## Architecture

```
Automation / Orchestration
    runner.py, report.py, claude_loop.py
    |
Evaluation & Metrics
    metrics.py, baselines.py, acceptance.py, machine_gun_proxy.py
    |
Model & Inference
    model.py, inference.py, sampling.py, train.py, config.py
    |
Data Pipeline
    dataset.py, alignment.py, preprocessing.py, codegram_cache.py, splits.py
    |
Codec (frozen, external)
    DAC 44.1 kHz pretrained
```

**Approach:** Frozen DAC neural audio codec as tokeniser/decoder + masked-token micro-inpainting transformer for variation generation + audio-domain acceptance filtering.

## Subsystems

| Subsystem | Status | Doc |
|-----------|--------|-----|
| Data Pipeline | Not started | — |
| Model & Inference | Not started | — |
| Evaluation & Metrics | Not started | — |
| Automation & Orchestration | Not started | — |

## Key Decisions

See [decisions/LOG.md](decisions/LOG.md) for the full decision log.

**Settled decisions (from research — do not revisit without evidence):**
1. Codec is DAC 44.1 kHz, frozen. No fine-tuning, no EnCodec.
2. Variation model is masked-token micro-inpainting. Not latent interpolation, not waveform diffusion.
3. Quality control is audio-domain acceptance filtering. Generate K candidates, decode, check perceptual distance.
4. Training objective is masked token cross-entropy. No backprop through DAC decoder for V1.
5. Editable codebooks use probabilistic gradient across 3-8. Not a hard cutoff.
6. Train at 1x mask rates, generate at 4x. Higher training mask rates kill variation.

## Open Questions

1. **Go/no-go feasibility** — The entire V1 exists to answer whether this approach works at all.
2. **Training data preparation** — Commercial round-robin libraries need processing into aligned codegram pairs.
3. **Acceptance filter thresholds** — Multi-metric thresholds for similarity/variation balance need calibration.
4. **Cloud training costs** — A100 time for training, L4/T4 for inference. Budget implications.

## Linked Projects

| Project | Relationship | Notes |
|---------|-------------|-------|
| AbletonSampler | related-to | Existing tooling for generating Ableton Sampler instrument files. Could consume VariationSampler output. |

## Notes

- V1 scope: snares first, expand to other drum families if initial results are promising.
- Output format: 44.1 kHz / 16-bit / stereo WAV.
- Perceptual quality is the ultimate authority — metrics serve the ear, never the other way around.
- Project structured for early, cheap failure via go/no-go gates.
- The project owner is a musician with direct domain expertise in drum samples.
- Training on commercial round-robin drum libraries (learns variation patterns, not specific sounds).
- No user-facing application, interface, or distribution mechanism in V1.

## Key Files

| File | Purpose |
|------|---------|
| `VisionStatement-PM.md` | Product vision |
| `CLAUDE.md` | Project conventions and architecture reference |
| `ledger/ROADMAP.md` | Development roadmap |
