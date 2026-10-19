# Batch Summary: phase4-002

## Metadata

- **Iterations:** 4
- **Stop reason:** stagnation
- **Start:** 2026-03-03T08:09:27.374931+00:00
- **End:** 2026-03-03T08:20:22.300427+00:00
- **Git commit:** `f32a03191f9fc80c6f1bfabc8a5817bbcf6818d3`

## Config: Start vs End

No config changes (same config start to finish).

**Starting config:**

- `k_candidates`: 16
- `mask_p_attack`: 0.02
- `mask_p_tail`: 0.24
- `temperature`: 0.65
- `top_p`: 0.9

## Metric Trajectory

| Metric | Trend | Iter 0 | Iter 1 | Iter 2 | Iter 3 |
|--------|-------| --- | --- | --- | --- |
| accepted | → | 0.4208 | 0.4042 | 0.4125 | 0.4083 |
| attack_smear | → | 0.9067 | 0.9104 | 0.9089 | 0.9097 |
| hf_energy_delta_db | → | -0.9996 | -1.0346 | -0.9967 | -1.0072 |
| inter_var_mrstft_mean | → | 0.6875 | 0.6852 | 0.6880 | 0.6836 |
| mfcc | → | 22.4462 | 22.6134 | 22.7576 | 22.6857 |
| mrstft | → | 1.3682 | 1.3671 | 1.3689 | 1.3682 |
| mrstft_attack | → | 1.1455 | 1.1453 | 1.1455 | 1.1439 |
| spectral_peak_divergence | → | 18.6500 | 18.6333 | 18.7417 | 18.5583 |
| token_change_rate | → | 0.1335 | 0.1341 | 0.1335 | 0.1341 |
| transient_xcorr | → | 0.7175 | 0.7189 | 0.7171 | 0.7187 |

## Per-Family Breakdown (Final Iteration)

| Family | N | MR-STFT (mean) | MFCC (mean) | Acceptance | vs Baseline |
|--------|---|----------------|-------------|------------|-------------|
| CrossStick | 3 | 1.572 | 34.3 | N/A | mrstft: OUT (bl median=1.023) mfcc: OUT (bl median=15.401) |
| HiHat | 3 | 1.868 | 31.0 | N/A | mrstft: OUT (bl median=1.017) mfcc: OUT (bl median=13.555) |
| Kick | 3 | 0.971 | 8.2 | N/A | mrstft: in-band (bl median=0.649) mfcc: in-band (bl median=5.134) |
| Rimshot | 3 | 1.286 | 16.7 | N/A | mrstft: in-band (bl median=1.026) mfcc: in-band (bl median=16.731) |
| Snare | 3 | 1.144 | 23.4 | N/A | mrstft: in-band (bl median=0.934) mfcc: OUT (bl median=13.218) |

## Best Samples

- **pass-02_Track11_Kick_v025** (score: 1.7902, iter: 1)
- **pass-02_Track12_Kick_v025** (score: 1.9464, iter: 0)
- **pass-02_Track10_Kick_v025** (score: 2.2278, iter: 0)
- **pass-02_Track14_CrossStick_v102** (score: 5.4660, iter: 1)
- **pass-02_Track11_SnareRim_v025** (score: 5.8084, iter: 0)

## Worst Samples

- **pass-02_Track12_Kick_v025** (score: 1.9464, iter: 0)
- **pass-02_Track10_Kick_v025** (score: 2.2278, iter: 0)
- **pass-02_Track14_CrossStick_v102** (score: 5.4660, iter: 1)
- **pass-02_Track11_SnareRim_v025** (score: 5.8084, iter: 0)
- **pass-02_Track11_CrossStick_v051** (score: 6.0943, iter: 1)

## Claude Diagnoses

### Iteration 1

The acceptance rate is ~40%, which is relatively low. Let me analyze the key metrics:

1. **token_change_rate** is only 0.134 (13.4%), which is quite low - variations aren't changing much from the source.
2. **mrstft** mean of 1.367 suggests moderate spectral distance, but many samples are being rejected.
3. **attack_smear** is 0.91 (close to 1.0 is good), meaning attacks are mostly preserved.
4. **transient_xcorr** mean is 0.719 but p50 is 0.891 - there's a tail of samples with poor transient correlation, likely causing rejections.
5. **hf_energy_delta_db** mean is -1.03 dB, indicating slight HF energy loss on average.
6. **mrstft_attack** mean of 1.145 is notable - attack regions are being distorted enough to cause rejections.

The low acceptance rate combined with low token_change_rate suggests that when tokens DO change, they're sometimes changing in destructive ways (bad transients, attack distortion). This points to temperature/top_p being configured such that token replacements are sometimes too radical even though they're infrequent.

Strategy:
- **Increase mask_p_tail slightly** (0.24 → 0.28): Allow more tail tokens to be masked, giving more opportunity for variation in the less-sensitive tail region.
- **Keep mask_p_attack low** (0.02 → 0.02): Attack preservation is important and attack_smear is decent, so don't touch this.
- **Lower temperature slightly** (0.65 → 0.55): Reduce the randomness of token selection to make replacements more conservative/likely to be accepted. This should improve acceptance rate.
- **Reduce top_p slightly** (0.9 → 0.85): Tighten the sampling distribution to avoid outlier tokens that cause transient degradation.
- **Increase k_candidates** (16 → 20): More candidates to select from should help find better variations that pass quality filters.

The net effect should be: more masking in the tail (more variation potential) but more conservative token selection (higher acceptance), with more candidates to pick from.

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
