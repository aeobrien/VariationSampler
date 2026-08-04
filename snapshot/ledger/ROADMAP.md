# Roadmap

## Next Up

| Task | Milestone | Phase | Status | Effort |
|------|-----------|-------|--------|--------|
| 1.1.1 Set up Python project structure, dependencies, configs | 1.1 Project Setup | 1: Data Pipeline | Todo | Administrative |
| 1.1.2 Set up Google Cloud environment (A100 training, L4/T4 inference) | 1.1 Project Setup | 1: Data Pipeline | Todo | Administrative |
| 1.2.1 Import and catalogue commercial round-robin drum libraries | 1.2 Data Import & Preprocessing | 1: Data Pipeline | Todo | Deep Focus |

---

## Phase 1: Data Pipeline
**Status:** Todo
**Definition of Done:** Round-robin drum libraries imported, preprocessed, aligned, encoded to DAC codegrams, cached, and split into train/dev/test sets. Procedural baseline (Gate 0) established.

### 1.1 — Project Setup
**Status:** Todo
**Priority:** High
**Definition of Done:** Repository structure, dependencies, cloud environment, and W&B integration ready.

| # | Task | Status | Effort | Deadline | Notes |
|---|------|--------|--------|----------|-------|
| 1.1.1 | Set up Python project structure, dependencies, configs | Todo | Administrative | | requirements.txt, configs/, src/ layout |
| 1.1.2 | Set up Google Cloud environment | Todo | Administrative | | A100 for training, L4/T4 for inference |
| 1.1.3 | Set up Weights & Biases integration | Todo | Administrative | | Experiment tracking |

### 1.2 — Data Import & Preprocessing
**Status:** Todo
**Priority:** High
**Definition of Done:** Raw round-robin sets imported, trimmed, aligned, normalised. Onset detection validated.

| # | Task | Status | Effort | Deadline | Notes |
|---|------|--------|--------|----------|-------|
| 1.2.1 | Import and catalogue commercial round-robin drum libraries | Todo | Deep Focus | | Start with snares |
| 1.2.2 | Implement onset detection and trimming | Todo | Deep Focus | | librosa-based |
| 1.2.3 | Implement round-robin alignment within sets | Todo | Deep Focus | | Transient alignment critical |
| 1.2.4 | Implement normalisation pipeline | Todo | Deep Focus | | 44.1 kHz, consistent levels |

### 1.3 — Codec Encoding & Splits
**Status:** Todo
**Priority:** High
**Definition of Done:** All preprocessed samples encoded to DAC codegrams, cached. Train/dev/test splits created with no library leakage across splits.

| # | Task | Status | Effort | Deadline | Notes |
|---|------|--------|--------|----------|-------|
| 1.3.1 | Implement DAC encoding pipeline | Todo | Deep Focus | | Cache codegrams, don't re-encode |
| 1.3.2 | Validate DAC reconstruction quality on source material | Todo | Deep Focus | | Confirm codec isn't the bottleneck |
| 1.3.3 | Create train/dev/test splits (no library leakage) | Todo | Deep Focus | | Split by library, not by sample |

### 1.4 — Gate 0: Procedural Baseline
**Status:** Todo
**Priority:** High
**Definition of Done:** Procedural variation baseline established (pitch shift, EQ, timing jitter). Metric distributions computed. This is the quality floor ML must beat.

| # | Task | Status | Effort | Deadline | Notes |
|---|------|--------|--------|----------|-------|
| 1.4.1 | Implement procedural variation generators | Todo | Deep Focus | | Pitch shift, micro-EQ, timing jitter |
| 1.4.2 | Compute baseline metric distributions | Todo | Deep Focus | | MR-STFT distance, MFCC distance against real RR sets |
| 1.4.3 | Document Gate 0 baseline results | Todo | Administrative | | Quality floor for ML comparison |

---

## Phase 2: Model & Training
**Status:** Todo
**Definition of Done:** Masked-token micro-inpainting transformer trained on snare round-robin sets. Model produces token-level variations. Gate A: first listenable outputs evaluated.

### 2.1 — Model Architecture
**Status:** Todo
**Priority:** High
**Definition of Done:** Transformer architecture implemented with masked-token prediction on codegrams. Probabilistic mask gradient across codebooks 3-8.

| # | Task | Status | Effort | Deadline | Notes |
|---|------|--------|--------|----------|-------|
| 2.1.1 | Implement codegram dataset and dataloader | Todo | Deep Focus | | Masked token pairs from RR sets |
| 2.1.2 | Implement micro-inpainting transformer architecture | Todo | Deep Focus | | Cross-entropy on masked tokens |
| 2.1.3 | Implement probabilistic mask gradient (codebooks 3-8) | Todo | Deep Focus | | Exponential: ~0.5% at cb3, ~15% at cb8 |
| 2.1.4 | Implement training loop with W&B logging | Todo | Deep Focus | | |

### 2.2 — Training & Gate A
**Status:** Todo
**Priority:** High
**Definition of Done:** Model trained. First listenable outputs produced and evaluated. Go/no-go decision on whether to continue.

| # | Task | Status | Effort | Deadline | Notes |
|---|------|--------|--------|----------|-------|
| 2.2.1 | Train model on snare round-robin sets (1x mask rates) | Todo | Deep Focus | | A100 training |
| 2.2.2 | Implement inference pipeline (4x mask rates) | Todo | Deep Focus | | Train 1x, infer 4x — critical |
| 2.2.3 | Generate first listenable outputs | Todo | Deep Focus | | |
| 2.2.4 | Gate A: human listening evaluation | Todo | Creative | | Go/no-go. Can we hear convincing variation? |

---

## Phase 3: Evaluation & Acceptance
**Status:** Todo
**Definition of Done:** Multi-metric evaluation pipeline working. Acceptance filter tuned. Variations pass machine-gun test and two-axis validation on snares.

### 3.1 — Metrics & Acceptance Filter
**Status:** Todo
**Priority:** High
**Definition of Done:** MR-STFT, MFCC distance, machine-gun proxy metric implemented. Acceptance filter produces outputs that pass human evaluation.

| # | Task | Status | Effort | Deadline | Notes |
|---|------|--------|--------|----------|-------|
| 3.1.1 | Implement multi-resolution STFT distance metric | Todo | Deep Focus | | |
| 3.1.2 | Implement MFCC distance metric | Todo | Deep Focus | | |
| 3.1.3 | Implement machine-gun proxy metric | Todo | Deep Focus | | Rapid retriggering test |
| 3.1.4 | Implement acceptance filter (generate K, keep best) | Todo | Deep Focus | | Audio-domain filtering, not token-space |
| 3.1.5 | Calibrate acceptance thresholds against real RR sets | Todo | Creative | | Thresholds from commercial library distributions |

### 3.2 — Validation
**Status:** Todo
**Priority:** High
**Definition of Done:** Variations pass machine-gun test, two-axis validation (similarity + variation), and cross-source validation on held-out libraries.

| # | Task | Status | Effort | Deadline | Notes |
|---|------|--------|--------|----------|-------|
| 3.2.1 | Machine-gun test: rapid sequence evaluation | Todo | Creative | | Expert listening, 16ths at 120 BPM |
| 3.2.2 | Two-axis validation: similarity and variation balance | Todo | Creative | | Same instrument + perceptibly different |
| 3.2.3 | Cross-source validation on held-out libraries | Todo | Deep Focus | | Model learned patterns, not studio fingerprints |
| 3.2.4 | Compare ML output to procedural baseline (Gate 0) | Todo | Deep Focus | | ML must beat the floor |

---

## Phase 4: Expansion & Polish
**Status:** Todo
**Definition of Done:** System works reliably across drum families (not just snares). Automation loop operational. V1 proof of concept complete.

### 4.1 — Instrument Expansion
**Status:** Todo
**Priority:** Normal
**Definition of Done:** Validated on kicks, hi-hats, toms, cymbals (beyond initial snare focus).

| # | Task | Status | Effort | Deadline | Notes |
|---|------|--------|--------|----------|-------|
| 4.1.1 | Expand training data to additional drum families | Todo | Deep Focus | | Only if snare results are promising |
| 4.1.2 | Evaluate cross-family performance | Todo | Creative | | May need per-family tuning |

### 4.2 — Automation Loop
**Status:** Todo
**Priority:** Normal
**Definition of Done:** Automated iteration loop: train, generate, evaluate, report. Claude returns config JSON with reasoning (no code edits in unattended loop).

| # | Task | Status | Effort | Deadline | Notes |
|---|------|--------|--------|----------|-------|
| 4.2.1 | Implement runner.py orchestration | Todo | Deep Focus | | Post-Gate A investment |
| 4.2.2 | Implement report.py summary generation | Todo | Deep Focus | | |
| 4.2.3 | Implement claude_loop.py (config-only, no code edits) | Todo | Deep Focus | | Claude suggests config changes only |

---

## Dependencies

| Item | Depends On | Status |
|------|-----------|--------|
| 2.1.1 (Dataset/dataloader) | 1.3 (Codec encoding & splits) | Unmet |
| 2.2.1 (Training) | 2.1 (Model architecture) | Unmet |
| 2.2.2 (Inference at 4x) | 2.2.1 (Trained model) | Unmet |
| 3.1.1-3.1.5 (Metrics) | 1.4 (Gate 0 baseline) | Unmet |
| 3.2.1-3.2.4 (Validation) | 2.2 (Gate A outputs) + 3.1 (Metrics) | Unmet |
| 4.1.1 (Expansion) | 3.2 (Validation on snares) | Unmet |
| 4.2.1-4.2.3 (Automation) | 2.2.4 (Gate A passed) | Unmet |

---

## Reference

### Status Values
| Status | Meaning |
|--------|---------|
| Todo | Not yet started |
| In Progress | Actively being worked on |
| Blocked: [reason] | Cannot proceed — reason is one of: poorly-defined, too-large, missing-info, missing-resource, decision-required |
| Waiting | User's part done, waiting on external input |
| Done | Complete |
| Dropped | Deliberately abandoned |

### Effort Types
| Type | Description |
|------|-------------|
| Deep Focus | Sustained concentration, problem-solving, design work |
| Creative | Open-ended, generative, exploratory |
| Administrative | Organising, documenting, updating, filing |
| Communication | Discussions, reviews, feedback |
| Physical | Hands-on work, building, soldering |
| Quick Win | Small, low-effort, momentum-building |

### Priority
High / Normal / Low — milestones only. Tasks inherit from their milestone unless overridden.
