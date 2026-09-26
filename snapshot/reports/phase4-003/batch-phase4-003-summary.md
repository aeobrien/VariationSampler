# Batch Summary: phase4-003

## Metadata

- **Iterations:** 6
- **Stop reason:** stagnation
- **Start:** 2026-03-03T08:36:49.571626+00:00
- **End:** 2026-03-03T09:03:04.459828+00:00
- **Git commit:** `f32a03191f9fc80c6f1bfabc8a5817bbcf6818d3`

## Config: Start vs End

| Parameter | Start | End |
|-----------|-------|-----|
| `k_candidates` | 16 | 28 |
| `mask_p_attack` | 0.02 | 0.0 |
| `mask_p_tail` | 0.24 | 0.18 |
| `temperature` | 0.65 | 0.35 |
| `top_p` | 0.9 | 0.75 |

**Starting config:**

- `k_candidates`: 16
- `mask_p_attack`: 0.02
- `mask_p_tail`: 0.24
- `temperature`: 0.65
- `top_p`: 0.9

## Config Trajectory (Per-Iteration)

| Iteration | Key | Old | New |
|-----------|-----|-----|-----|
| 1 | `k_candidates` | 16 | 20 |
| 1 | `mask_p_tail` | 0.24 | 0.22 |
| 1 | `temperature` | 0.65 | 0.55 |
| 1 | `top_p` | 0.9 | 0.85 |
| 3 | `k_candidates` | 20 | 24 |
| 3 | `mask_p_tail` | 0.22 | 0.18 |
| 3 | `temperature` | 0.55 | 0.45 |
| 3 | `top_p` | 0.85 | 0.8 |
| 5 | `k_candidates` | 24 | 28 |
| 5 | `mask_p_attack` | 0.02 | 0.0 |
| 5 | `temperature` | 0.45 | 0.35 |
| 5 | `top_p` | 0.8 | 0.75 |

## Metric Trajectory

| Metric | Trend | Iter 0 | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 |
|--------|-------| --- | --- | --- | --- | --- | --- |
| accepted | → | 0.4208 | 0.4033 | 0.4000 | 0.4028 | 0.4167 | 0.4024 |
| attack_smear | → | 0.9112 | 0.9068 | 0.9097 | 0.9076 | 0.9098 | 0.9090 |
| hf_energy_delta_db | → | -1.0282 | -1.0083 | -1.0303 | -1.0530 | -1.0345 | -1.0551 |
| inter_var_mrstft_mean | ↓ | 0.6875 | 0.6623 | 0.6614 | 0.6096 | 0.6090 | 0.5892 |
| mfcc | → | 22.7973 | 22.4457 | 22.2181 | 21.5765 | 21.8029 | 21.6184 |
| mrstft | → | 1.3673 | 1.3645 | 1.3634 | 1.3565 | 1.3571 | 1.3553 |
| mrstft_attack | → | 1.1436 | 1.1430 | 1.1415 | 1.1372 | 1.1382 | 1.1345 |
| spectral_peak_divergence | → | 18.5375 | 18.6067 | 18.2467 | 18.4889 | 18.3694 | 18.3143 |
| token_change_rate | ↓ | 0.1326 | 0.1210 | 0.1213 | 0.0983 | 0.0976 | 0.0975 |
| transient_xcorr | → | 0.7186 | 0.7171 | 0.7176 | 0.7191 | 0.7190 | 0.7219 |

## Per-Family Breakdown (Final Iteration)

| Family | N | MR-STFT (mean) | MFCC (mean) | Acceptance | vs Baseline |
|--------|---|----------------|-------------|------------|-------------|
| CrossStick | 3 | 1.553 | 32.9 | N/A | mrstft: in-band (bl median=1.023) mfcc: OUT (bl median=15.401) |
| HiHat | 3 | 1.864 | 30.2 | N/A | mrstft: OUT (bl median=1.017) mfcc: OUT (bl median=13.555) |
| Kick | 3 | 0.960 | 7.3 | N/A | mrstft: in-band (bl median=0.649) mfcc: in-band (bl median=5.134) |
| Rimshot | 3 | 1.267 | 15.5 | N/A | mrstft: in-band (bl median=1.026) mfcc: in-band (bl median=16.731) |
| Snare | 3 | 1.131 | 22.2 | N/A | mrstft: in-band (bl median=0.934) mfcc: in-band (bl median=13.218) |

## Best Samples

- **pass-02_Track11_Kick_v025** (score: 1.7026, iter: 5)
- **pass-02_Track12_Kick_v025** (score: 1.9005, iter: 5)
- **pass-02_Track10_Kick_v025** (score: 2.1590, iter: 3)
- **pass-02_Track14_CrossStick_v102** (score: 5.3713, iter: 2)
- **pass-02_Track11_SnareRim_v025** (score: 5.6965, iter: 3)

## Worst Samples

- **pass-02_Track12_Kick_v025** (score: 1.9005, iter: 5)
- **pass-02_Track10_Kick_v025** (score: 2.1590, iter: 3)
- **pass-02_Track14_CrossStick_v102** (score: 5.3713, iter: 2)
- **pass-02_Track11_SnareRim_v025** (score: 5.6965, iter: 3)
- **pass-02_Track11_CrossStick_v051** (score: 6.0232, iter: 1)

## Claude Diagnoses

### Iteration 0

First iteration with a 42% acceptance rate — there's significant room for improvement. Key observations:

1. **Transient preservation has outlier issues**: transient_xcorr mean (0.72) is much lower than p50 (0.89), indicating a subset of samples with badly damaged transients are dragging down acceptance.
2. **mrstft at 1.37 is moderate** but combined with spectral_peak_divergence of 18.5 and MFCC of 22.8, some variations may be diverging too far.
3. **token_change_rate is 0.13** — relatively low, so the problem isn't excessive masking but rather the *quality* of regenerated tokens in some cases.
4. **hf_energy_delta_db mean of -1.03** suggests slight HF loss, likely from imprecise token regeneration.

Strategy: Improve acceptance by increasing selection pressure (more candidates) and tightening generation quality (lower temperature, tighter nucleus). Slightly reduce mask_p_tail to keep variations closer to source and reduce chance of quality degradation. Keep mask_p_attack unchanged at 0.02 since it's already very conservative.

- k_candidates 16→20: More candidates to select from, better chance of filtering out bad outliers
- temperature 0.65→0.55: More peaked distributions reduce extreme/bad token choices
- top_p 0.9→0.85: Tighter nucleus cuts low-probability tail tokens that cause artifacts
- mask_p_tail 0.24→0.22: Slightly less masking to reduce regeneration errors while maintaining variation

### Iteration 2

Acceptance rate has been hovering around 40% for three iterations with all metrics stable — the current config is not converging toward improvement. The high mrstft (1.36), spectral_peak_divergence (18.25), and mfcc distance (22.2) indicate variations are deviating too far from the source spectrally, which is likely driving rejections. Meanwhile, transient_xcorr mean (0.72) is dragged down by a long tail of poor transient preservation despite a reasonable p50 (0.88). Token change rate is only ~12%, yet spectral impact is large, suggesting the replacement tokens themselves are too adventurous. Strategy: (1) Reduce mask_p_tail from 0.22→0.18 to reduce overall masking and constrain deviation; (2) Lower temperature from 0.55→0.45 so replacement tokens are more conservative/likely; (3) Lower top_p from 0.85→0.80 to further restrict sampling to high-probability tokens; (4) Increase k_candidates from 20→24 to give the selection stage more options to find high-quality variations. This multi-pronged approach should meaningfully reduce spectral distance metrics and boost acceptance rate while still preserving enough variation (inter_var_mrstft is currently 0.66, so we have room to trade some diversity for quality).

### Iteration 3

Acceptance rate has been flat around 40% for three iterations despite a low token change rate (~10%), indicating that the tokens being replaced are sometimes producing destructive results rather than there being too many changes. The bimodal transient_xcorr distribution (mean=0.72 vs p50=0.89) shows some samples are badly damaged while others are fine. The strategy should focus on improving the *quality* of token replacements rather than adjusting their quantity. I'll increase k_candidates to the maximum step (+4 → 28) to give the candidate selection process more options to find faithful variations. I'll lower temperature from 0.45 → 0.3 and tighten top_p from 0.8 → 0.7 to constrain sampling toward higher-probability tokens, reducing the chance of generating destructive replacements. mask_p_tail and mask_p_attack are left unchanged since token_change_rate and inter_var_mrstft are both trending in the right direction, and attack preservation (mask_p_attack=0.02) is already aggressive.

### Iteration 4

Acceptance rate has plateaued around 40-42% for 4 iterations with all metrics stable. Key observations: (1) Transient xcorr has a bimodal distribution (mean 0.72, p50 0.89) suggesting some samples have severely degraded transients, likely driving rejections. (2) Token change rate is low (~10%) yet acceptance is still poor, meaning the changes being made are sometimes too disruptive. (3) HF energy shows slight loss (-1.03 dB mean). Strategy: Fully eliminate attack masking (0.02→0.0) to maximally preserve transients. Increase k_candidates (24→28) for better candidate selection. Reduce temperature (0.45→0.35) and top_p (0.8→0.75) to make regenerated tokens more conservative—each changed token will be closer to the model's top predictions, reducing the chance of destructive edits. The tail mask stays the same so variation still comes from the tail region, but individual token replacements will be higher quality.

### Iteration 5

The system has been stuck at ~40% acceptance for 3 iterations with all metrics stable. The token_change_rate of only ~10% despite mask_p_tail=0.18 indicates the model is frequently re-predicting the original token, meaning we're not actually exploring the variation space effectively. The bimodal transient_xcorr distribution (mean=0.72, p50=0.89) suggests a subset of samples are degrading transients significantly while others are fine.

Strategy: Break out of the plateau by increasing exploration on multiple axes simultaneously. (1) Raise mask_p_tail from 0.18→0.23 to mask more tail tokens, creating more opportunity for meaningful variation. (2) Raise temperature from 0.35→0.50 so when tokens ARE masked, the model is more likely to actually substitute a different token rather than re-predicting the original — this should push token_change_rate up and improve diversity. (3) Lower top_p from 0.75→0.70 to compensate for the higher temperature by trimming the tail of the sampling distribution, avoiding low-probability garbage tokens. (4) Increase k_candidates from 28→32 to give the selection process more candidates to choose from, offsetting the potentially higher per-candidate rejection rate from increased exploration.

## Listening Notes

_To be filled in after auditioning the listening pack._

### Machine Gun Test

Compare `machinegun_source.wav` (same hit repeated) vs `machinegun_variations.wav` (ML variations):

| Sample | Source MG | Variations MG | Notes |
|--------|----------|---------------|-------|
| pass-02_Track11_Kick_v025 | | | |
| pass-02_Track12_Kick_v025 | | | |
| pass-02_Track10_Kick_v025 | | | |
| pass-02_Track14_CrossStick_v102 | | | |
| pass-02_Track11_SnareRim_v025 | | | |
| pass-02_Track11_CrossStick_v051 | | | |

### Overall Impressions

- Identity preservation: 
- Variation quality: 
- Problem families: 
- Recommended next step: 
