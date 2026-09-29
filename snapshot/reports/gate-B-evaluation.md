# Gate B Evaluation Report

## Summary

- **Date**: 2026-03-03T20:32:50.483213
- **Checkpoint**: `checkpoints/best.pt`
- **Total samples**: 50
- **Elapsed time**: 438.9s
- **Mask p_tail**: 0.64
- **Temperature**: 2.0

---

## Overall Metric Comparison

| Metric | ML (median) | Proc (median) | Real RR (median) |
|--------|-------------|---------------|------------------|
| MR-STFT | 1.5770 | 0.5445 | 0.9472 |
| MFCC | 26.9844 | 9.7025 | 12.6760 |
| Attack Smear | 0.5289 | 0.9157 | — |
| Transient Xcorr | 0.7586 | 0.2967 | — |
| HF Energy Delta (dB) | -2.9523 | -0.7311 | — |
| Spectral Peak Div. | 9.9167 | 2.0833 | — |
| Token Change Rate | 0.2734 | — | — |
| Inter-var MR-STFT | 0.9861 | 0.4457 | — |

## Machine-Gun Proxy Scores (spectral distance, higher = more variation)

| Family | ML | Procedural | Real RR |
|--------|----|------------|---------|
| CrossStick | 0.5517 | 0.5898 | 0.6271 |
| HiHat | 0.6648 | 0.6248 | 0.5962 |
| Kick | 0.5124 | 0.6857 | 0.5566 |
| Rimshot | 0.5857 | 0.5989 | 0.6446 |
| Snare | 0.6461 | 0.6171 | 0.6248 |
| **Overall** | **0.5861** | **0.6185** | **0.6057** |

## Acceptance Rate

| Family | Acceptance Rate (median) | n |
|--------|-------------------------|---|
| CrossStick | 0.00% | 10 |
| HiHat | 0.00% | 10 |
| Kick | 0.00% | 10 |
| Rimshot | 0.00% | 10 |
| Snare | 0.00% | 10 |
| **Overall** | **0.00%** | **50** |

## Per-Family Breakdown

### CrossStick (n=10)

| Metric | ML (median) | Proc (median) | Real RR (median) |
|--------|-------------|---------------|------------------|
| MR-STFT | 1.8158 | 0.5344 | 1.0973 |
| MFCC | 36.2040 | 7.6908 | 15.3678 |
| Attack Smear | 0.4124 | 0.9198 | — |
| Transient Xcorr | 0.5813 | 0.3025 | — |
| HF Energy Delta (dB) | -4.0196 | -0.6788 | — |
| Spectral Peak Div. | 6.9167 | 1.4167 | — |
| Token Change Rate | 0.2505 | — | — |
| Inter-var MR-STFT | 1.0333 | 0.4272 | — |

Machine-gun spectral distance: ML=0.5517, Proc=0.5898, RR=0.6271

In-band: {'mrstft': False, 'mfcc': False}

### HiHat (n=10)

| Metric | ML (median) | Proc (median) | Real RR (median) |
|--------|-------------|---------------|------------------|
| MR-STFT | 1.8985 | 0.5545 | 0.9924 |
| MFCC | 25.8299 | 10.5091 | 11.6402 |
| Attack Smear | 0.4661 | 0.8784 | — |
| Transient Xcorr | 0.3692 | 0.0866 | — |
| HF Energy Delta (dB) | -2.6889 | -1.1097 | — |
| Spectral Peak Div. | 14.7500 | 4.4167 | — |
| Token Change Rate | 0.1833 | — | — |
| Inter-var MR-STFT | 1.0468 | 0.4256 | — |

Machine-gun spectral distance: ML=0.6648, Proc=0.6248, RR=0.5962

In-band: {'mrstft': False, 'mfcc': True}

### Kick (n=10)

| Metric | ML (median) | Proc (median) | Real RR (median) |
|--------|-------------|---------------|------------------|
| MR-STFT | 1.2575 | 0.4933 | 0.6949 |
| MFCC | 13.4421 | 8.5170 | 4.9945 |
| Attack Smear | 0.9736 | 0.9268 | — |
| Transient Xcorr | 0.9863 | 0.8572 | — |
| HF Energy Delta (dB) | -0.3338 | -0.4603 | — |
| Spectral Peak Div. | 15.0833 | 0.6667 | — |
| Token Change Rate | 0.2573 | — | — |
| Inter-var MR-STFT | 0.7834 | 0.4305 | — |

Machine-gun spectral distance: ML=0.5124, Proc=0.6857, RR=0.5566

In-band: {'mrstft': False, 'mfcc': False}

### Rimshot (n=10)

| Metric | ML (median) | Proc (median) | Real RR (median) |
|--------|-------------|---------------|------------------|
| MR-STFT | 1.5535 | 0.5445 | 1.0395 |
| MFCC | 31.6325 | 10.6917 | 16.4274 |
| Attack Smear | 0.5892 | 0.9262 | — |
| Transient Xcorr | 0.8387 | 0.4546 | — |
| HF Energy Delta (dB) | -2.8441 | -0.7224 | — |
| Spectral Peak Div. | 5.5000 | 1.9167 | — |
| Token Change Rate | 0.3054 | — | — |
| Inter-var MR-STFT | 1.0048 | 0.4598 | — |

Machine-gun spectral distance: ML=0.5857, Proc=0.5989, RR=0.6446

In-band: {'mrstft': False, 'mfcc': False}

### Snare (n=10)

| Metric | ML (median) | Proc (median) | Real RR (median) |
|--------|-------------|---------------|------------------|
| MR-STFT | 1.3906 | 0.6145 | 0.9759 |
| MFCC | 26.3185 | 9.6597 | 14.0467 |
| Attack Smear | 0.3937 | 0.9207 | — |
| Transient Xcorr | 0.5939 | 0.0758 | — |
| HF Energy Delta (dB) | -3.6855 | -0.9013 | — |
| Spectral Peak Div. | 6.9167 | 2.4167 | — |
| Token Change Rate | 0.3202 | — | — |
| Inter-var MR-STFT | 0.9763 | 0.4788 | — |

Machine-gun spectral distance: ML=0.6461, Proc=0.6171, RR=0.6248

In-band: {'mrstft': True, 'mfcc': False}

---

## Listening Assessment

*(To be filled in by project owner after listening to `machine_gun_ab/` directory)*

### Machine-gun test (does the ML output break repetition?)

| Family | ML breaks repetition? | ML vs Proc | ML vs Real RR | Notes |
|--------|-----------------------|------------|---------------|-------|
| CrossStick | [ ] Yes / [ ] No | [ ] Better / [ ] Same / [ ] Worse | [ ] Close / [ ] Gap | |
| HiHat | [ ] Yes / [ ] No | [ ] Better / [ ] Same / [ ] Worse | [ ] Close / [ ] Gap | |
| Kick | [ ] Yes / [ ] No | [ ] Better / [ ] Same / [ ] Worse | [ ] Close / [ ] Gap | |
| Rimshot | [ ] Yes / [ ] No | [ ] Better / [ ] Same / [ ] Worse | [ ] Close / [ ] Gap | |
| Snare | [ ] Yes / [ ] No | [ ] Better / [ ] Same / [ ] Worse | [ ] Close / [ ] Gap | |

### Overall quality

- [ ] Variations sound like the same instrument struck again
- [ ] Variations sound different from the source (not identical copies)
- [ ] No audible artifacts (clicks, tonal smearing, HF loss)
- [ ] Attack transients are preserved
- [ ] ML output consistently beats procedural baseline

### Any samples that stood out (good or bad)?

*(notes here)*

---

## Decision

- [ ] **PASS** — ML variations match real RR magnitude and beat procedural baseline
- [ ] **ADJUST** — Promising but needs parameter tuning (specify what)
- [ ] **FAIL** — Fundamental issues prevent Gate B passage

**Signed**: _________________________ **Date**: _____________
