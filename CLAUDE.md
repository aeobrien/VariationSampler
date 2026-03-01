# Variation Sampler — CLAUDE.md

This file is read by every Claude Code instance working on this project. It defines the project overview, architecture, conventions, and institutional knowledge.

---

## Overview

Variation Sampler is a machine learning system that generates perceptually convincing round-robin variations from a single one-shot drum sample. Given one recording of a drum hit, it produces 4–8 variations that sound like the same instrument struck again — not like the same recording replayed, and not like the original has been processed.

V1 is a proof of concept. The sole question: can an ML model generate drum sample variations that preserve source identity while breaking perceptual repetition?

The project owner is a musician with deep domain expertise in drum samples. Their perceptual judgement is the ultimate authority — it overrides all quantitative metrics.

---

## Architecture

**Approach:** Frozen neural audio codec (DAC) as tokeniser/decoder + masked-token micro-inpainting transformer for variation generation + audio-domain acceptance filtering.

**Key insight:** Codec token spaces are not perceptually smooth. Token indices are categorical — entry 123 is not "closer to" 124 than to 900. The micro-inpainting approach works by predicting what tokens would plausibly appear if this were a different recording of the same hit. The "smallness prior" is structural (sparse edits in late codebooks), not numeric (no Euclidean smoothness assumption).

**Stack:**
- Python 3.10+
- PyTorch (training, inference, model code)
- DAC (Descript Audio Codec) — frozen, pretrained 44.1 kHz weights
- librosa (onset detection, spectral analysis, preprocessing)
- Weights & Biases (experiment tracking)
- Google Cloud (A100 for training, L4/T4 for inference/prototyping)

**Components (in dependency order):**

```
┌──────────────────────────────────────────────────┐
│              Automation / Orchestration            │
│  runner.py · report.py · claude_loop.py           │
└────────────────────────┬─────────────────────────┘
                         │ uses
┌────────────────────────┴─────────────────────────┐
│              Evaluation & Metrics                  │
│  metrics.py · baselines.py · acceptance.py        │
│  machine_gun_proxy.py                             │
└────────────────────────┬─────────────────────────┘
                         │ uses
┌────────────────────────┴─────────────────────────┐
│              Model & Inference                     │
│  model.py · inference.py · sampling.py            │
│  train.py · config.py                             │
└────────────────────────┬─────────────────────────┘
                         │ uses
┌────────────────────────┴─────────────────────────┐
│              Data Pipeline                         │
│  dataset.py · alignment.py · preprocessing.py     │
│  codegram_cache.py · splits.py                    │
└────────────────────────┬─────────────────────────┘
                         │ uses
┌────────────────────────┴─────────────────────────┐
│              Codec (frozen, external)              │
│  DAC 44.1 kHz pretrained                          │
└──────────────────────────────────────────────────┘
```

---

## DAC Codec Reference

These are constants. Do not change unless switching codec.

| Parameter | Value |
|---|---|
| Sample rate | 44,100 Hz |
| RVQ codebooks (Nq) | 9 |
| Codebook size (V) | 1,024 entries |
| Stride | 512 samples |
| Latent frame rate | 86 Hz |
| Token throughput | 774 tokens/sec |
| Codegram shape (1s audio) | `int[9, 86]`, values 0–1023 |
| Max audio length (V1) | 1.0 second |

---

## Project Structure

```
VariationSampler/
├── CLAUDE.md                    # This file
├── ROADMAP.md                   # Phased development plan
├── WORKFLOW.md                  # Development process contract
├── docs/
│   ├── vision-statement.md      # Project vision and motivation
│   ├── technical-brief.md       # Full technical specification
│   ├── research-report-1.md     # Feasibility research
│   ├── research-report-2.md     # Implementation-depth research
│   ├── implementation-notes.md  # Practical guidance and caveats
│   ├── session-log.md           # Session history
│   └── listening-notes/         # Human listening check-in notes
├── src/
│   ├── data/                    # Data pipeline, preprocessing, alignment
│   ├── model/                   # Model architecture, training, inference
│   ├── eval/                    # Metrics, baselines, acceptance filtering
│   ├── automation/              # Runner, reporting, Claude API loop
│   ├── postprocess/             # Post-processing chain (DC removal, dither, etc.)
│   └── utils/                   # Shared utilities
├── configs/                     # YAML/JSON experiment configs
├── scripts/                     # One-off scripts (data import, codec testing, etc.)
├── tests/                       # Test suite
├── data/                        # Local data directory (gitignored)
│   ├── raw/                     # Imported WAV files
│   ├── processed/               # Trimmed, aligned, normalised
│   ├── codegrams/               # Cached DAC encodings
│   ├── baselines/               # Ground-truth metric distributions
│   └── splits/                  # Train/dev/test split manifests
├── outputs/                     # Generated variations (gitignored)
├── reports/                     # Iteration reports (JSON + summary)
└── requirements.txt             # Python dependencies
```

---

## Code Style

### Python Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Files | snake_case | `codegram_cache.py` |
| Classes | PascalCase | `VariationTransformer` |
| Functions/methods | snake_case | `compute_mrstft_distance()` |
| Constants | UPPER_SNAKE_CASE | `MAX_AUDIO_LENGTH_S = 1.0` |
| Config keys | snake_case | `mask_p_tail: 0.08` |
| Test files | `test_` prefix | `test_alignment.py` |
| Test functions | `test_` prefix, descriptive | `test_alignment_preserves_transient()` |

### Code Organisation

- One module per concern. Don't combine unrelated functionality.
- Configs are YAML/JSON files in `configs/`, not hardcoded values.
- All hyperparameters that the automation loop might tune must be in config, not in code.
- Type hints on all function signatures.
- Docstrings on public functions — brief, focused on what and why, not how.

### Logging

Use Python `logging` module. One logger per module.

```python
import logging
logger = logging.getLogger(__name__)
```

| Level | When |
|-------|------|
| `DEBUG` | Internal state, tensor shapes, intermediate values |
| `INFO` | Normal operations: "Loaded 1200 codegrams", "Training epoch 5 complete" |
| `WARNING` | Non-fatal issues: low acceptance rate, missing metadata |
| `ERROR` | Recoverable failures: failed to decode sample, API timeout |
| `CRITICAL` | Unrecoverable: corrupt data, GPU OOM |

`print()` is banned. Use `logger` for all output.

### Audio Conventions

- All internal audio processing in float32
- Sample rate: 44,100 Hz always. Never resample silently.
- Channel order: `[channels, samples]` for raw audio tensors
- Codegram shape: `[Nq, T]` for a single sample, `[B, Nq, T]` for batched
- WAV I/O: use `soundfile` or `torchaudio`. Always verify sample rate on load.
- Never overwrite source audio files. All outputs go to `outputs/`.

---

## Testing Conventions

**Framework:** pytest

**Rules:**
- Every public function in `src/` gets at least one test
- Test data: small synthetic fixtures, not full audio files (keep tests fast)
- Codec integration tests (that need DAC weights) are marked `@pytest.mark.slow` and excluded from quick runs
- GPU-dependent tests are marked `@pytest.mark.gpu`
- No test should depend on another test's state
- Use `tmp_path` fixture for any file I/O in tests

```bash
# Fast tests (no GPU, no codec)
pytest tests/ -m "not slow and not gpu"

# All tests
pytest tests/

# Specific module
pytest tests/test_alignment.py -v
```

---

## Key Reference Documents

When making decisions, consult:

1. **Technical Brief** (`docs/technical-brief.md`) — Full specification: architecture, data pipeline, metrics, evaluation, automation loop, go/no-go gates. This is the authoritative technical reference.
2. **Vision Statement** (`docs/vision-statement.md`) — Product intent, design principles, definition of done. Takes precedence on "what should this sound like" questions.
3. **Implementation Notes** (`docs/implementation-notes.md`) — Practical caveats, Goodhart's Law warnings, cost estimates, confidence assessment.
4. **Research Reports** (`docs/research-report-1.md`, `docs/research-report-2.md`) — Architecture rationale, prior art, codec comparison, latent space analysis.
5. **ROADMAP.md** — Phase sequencing and deliverables.
6. **WORKFLOW.md** — Process contract: how development sessions and automation loops work.

The vision statement takes precedence on perceptual quality questions. The technical brief takes precedence on implementation details. When they conflict, flag it.

---

## Critical Design Decisions (Do Not Revisit Without Evidence)

These are settled for V1 based on two rounds of research:

1. **Codec is DAC 44.1 kHz, frozen.** Do not fine-tune. Do not switch to EnCodec. Only revisit if reconstruction quality on source material is demonstrably the bottleneck.

2. **Variation model is masked-token micro-inpainting.** Not latent interpolation, not Gaussian noise on embeddings, not waveform diffusion. The categorical nature of codec tokens makes smooth-delta approaches unreliable.

3. **Quality control is audio-domain acceptance filtering.** Generate K candidates, decode, check perceptual distance, keep or reject. This is the primary guardrail. Do not rely on token-space distance as a proxy for perceptual distance.

4. **Training objective is masked token cross-entropy.** No need to backpropagate through the DAC decoder for V1. Waveform losses are a V1.5 consideration only if token CE + acceptance filtering proves insufficient.

5. **Editable codebooks use a probabilistic gradient across 3–8.** Not a hard cutoff. Earlier codebooks get exponentially lower mask probability (e.g., codebook 3 at ~0.5%, codebook 8 at ~15%). The acceptance filter is the ultimate guardrail. Start conservatively and tune the gradient based on listening.

---

## Mistakes to Avoid

*This section is updated as the project progresses. Add entries here when something goes wrong that could go wrong again.*

- **[Architecture] Do not interpolate in token space.** Token indices are categorical. "Entry 123 + small delta" is meaningless. The micro-inpainting approach exists specifically to avoid this.
- **[Architecture] Do not use a hard codebook cutoff.** Use a probabilistic gradient across codebooks 3–8, not a hard restriction to 6–8. Real RR variation includes spectral envelope changes in earlier codebooks, not just fine texture.
- **[Data] Do not include cross-velocity pairs in training.** V1 learns micro-variation at constant intensity. Cross-velocity pairs teach the model to change dynamics, which contaminates the learning target.
- **[Data] Do not re-encode audio through DAC on every training run.** Cache codegrams once as a preprocessing step. Re-encoding wastes GPU time and adds noise.
- **[Metrics] Do not optimise a single metric.** Any single metric can be gamed (broadband noise, inaudible HF changes, timing shifts). Always evaluate multi-metric + listening.
- **[Metrics] Always compare ML output to the procedural baseline.** The procedural baseline from Gate 0 is the quality floor. If ML can't beat it, ML isn't adding value.
- **[Automation] Do not let Claude edit code in the automated loop.** Claude's role in the unattended loop is returning config JSON with reasoning only. Code changes happen in supervised sessions.
- **[Automation] Do not build the automation loop before Gate A.** Manual iteration is faster for reaching first listenable outputs. Automation is a post-Gate A investment.
