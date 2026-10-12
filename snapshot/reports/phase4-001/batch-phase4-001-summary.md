# Batch Summary: phase4-001

## Metadata

- **Iterations:** 6
- **Stop reason:** stagnation
- **Start:** 2026-03-02T23:20:01.481547+00:00
- **End:** 2026-03-02T23:32:52.854263+00:00
- **Git commit:** `f32a03191f9fc80c6f1bfabc8a5817bbcf6818d3`

## Config: Start vs End

| Parameter | Start | End |
|-----------|-------|-----|
| `k_candidates` | 8 | 16 |
| `mask_p_tail` | 0.32 | 0.24 |
| `temperature` | 0.9 | 0.65 |
| `top_p` | 0.95 | 0.9 |

**Starting config:**

- `k_candidates`: 8
- `mask_p_attack`: 0.08
- `mask_p_tail`: 0.32
- `temperature`: 0.9
- `top_p`: 0.95

## Config Trajectory (Per-Iteration)

| Iteration | Key | Old | New |
|-----------|-----|-----|-----|
| 2 | `k_candidates` | 8 | 12 |
| 2 | `mask_p_tail` | 0.32 | 0.28 |
| 2 | `temperature` | 0.9 | 0.75 |
| 2 | `top_p` | 0.95 | 0.9 |
| 3 | `k_candidates` | 12 | 16 |
| 3 | `mask_p_tail` | 0.28 | 0.24 |
| 3 | `temperature` | 0.75 | 0.65 |

## Metric Trajectory

| Metric | Trend | Iter 0 | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 |
|--------|-------| --- | --- | --- | --- | --- | --- |
| accepted | → | 0.5083 | 0.5167 | 0.5333 | 0.5333 | 0.5292 | 0.5250 |
| attack_smear | → | 0.9601 | 0.9598 | 0.9600 | 0.9606 | 0.9613 | 0.9596 |
| hf_energy_delta_db | → | -0.7419 | -0.7240 | -0.6751 | -0.6991 | -0.6798 | -0.7104 |
| inter_var_mrstft_mean | ↓ | 0.8245 | 0.8122 | 0.7818 | 0.7331 | 0.7369 | 0.7353 |
| mfcc | → | 25.4771 | 25.7833 | 25.0563 | 24.6336 | 24.5349 | 24.4643 |
| mrstft | → | 1.4255 | 1.4189 | 1.4093 | 1.4022 | 1.4010 | 1.3999 |
| mrstft_attack | → | 1.1867 | 1.1787 | 1.1783 | 1.1686 | 1.1697 | 1.1699 |
| spectral_peak_divergence | → | 20.4750 | 20.8000 | 20.6389 | 20.5542 | 20.6708 | 20.3458 |
| token_change_rate | ↓ | 0.1848 | 0.1864 | 0.1617 | 0.1373 | 0.1360 | 0.1361 |
| transient_xcorr | → | 0.6952 | 0.7024 | 0.7014 | 0.7057 | 0.7024 | 0.7039 |

## Per-Family Breakdown (Final Iteration)

| Family | N | MR-STFT (mean) | MFCC (mean) | Acceptance | vs Baseline |
|--------|---|----------------|-------------|------------|-------------|
| CrossStick | 3 | 1.619 | 35.2 | N/A | mrstft: OUT (bl median=1.023) mfcc: OUT (bl median=15.401) |
| HiHat | 3 | 1.894 | 33.8 | N/A | mrstft: OUT (bl median=1.017) mfcc: OUT (bl median=13.555) |
| Kick | 3 | 0.982 | 8.3 | N/A | mrstft: in-band (bl median=0.649) mfcc: in-band (bl median=5.134) |
| Rimshot | 3 | 1.337 | 18.8 | N/A | mrstft: in-band (bl median=1.026) mfcc: in-band (bl median=16.731) |
| Snare | 3 | 1.167 | 26.2 | N/A | mrstft: in-band (bl median=0.934) mfcc: OUT (bl median=13.218) |

## Best Samples

- **pass-02_Track11_Kick_v025** (score: 1.8177, iter: 3)
- **pass-02_Track12_Kick_v025** (score: 2.0297, iter: 5)
- **pass-02_Track10_Kick_v025** (score: 2.2365, iter: 3)
- **pass-02_Track12_HiHatClosed_v076** (score: 5.8003, iter: 4)
- **pass-02_Track11_SnareRim_v025** (score: 6.3210, iter: 3)

## Worst Samples

- **pass-02_Track12_Kick_v025** (score: 2.0297, iter: 5)
- **pass-02_Track10_Kick_v025** (score: 2.2365, iter: 3)
- **pass-02_Track12_HiHatClosed_v076** (score: 5.8003, iter: 4)
- **pass-02_Track11_SnareRim_v025** (score: 6.3210, iter: 3)
- **pass-02_Track11_CrossStick_v051** (score: 6.4076, iter: 5)

## Claude Diagnoses

### Iteration 0

This is the first iteration, so I'm analyzing the baseline metrics to identify issues:

1. **Acceptance rate is low (0.508)** - only about half the generated variations are being accepted, meaning many fail quality thresholds.

2. **transient_xcorr is concerning** - mean=0.695 but p50=0.869 suggests a bimodal distribution where some samples have very poor transient preservation. The attack_smear mean=0.96 (close to 1.0) is decent but the spread matters.

3. **mrstft is relatively high (1.43)** - this suggests variations are deviating quite a bit from the source spectrally. Combined with the low acceptance rate, many variations are likely too different.

4. **token_change_rate is moderate (0.185)** - about 18.5% of tokens are changed, which seems reasonable but given the high mrstft and low acceptance, the changes may be happening in perceptually sensitive areas.

5. **hf_energy_delta_db mean=-0.74** - slight loss of high-frequency energy on average, though p50 is near 0. Some samples are losing significant HF content.

6. **mrstft_attack is high (1.19)** - attack regions are being modified too much, which explains poor transient_xcorr for some samples.

**Strategy:** The main issues are too-aggressive modification leading to rejected samples. I should:
- **Reduce mask_p_attack** from 0.08 to 0.06 to better preserve transients/attacks, which should improve transient_xcorr and mrstft_attack.
- **Reduce mask_p_tail** from 0.32 to 0.28 to reduce overall variation magnitude, improving mrstft and acceptance rate.
- **Reduce temperature** from 0.9 to 0.8 to make token replacements more conservative/likely, reducing wild deviations.
- **Keep top_p at 0.95** - already reasonable for diversity.
- **Increase k_candidates** from 8 to 12 to give the selection algorithm more candidates to choose from, improving the quality of accepted samples.

### Iteration 1

Analyzing the metrics:

1. **Acceptance rate is ~52%**, which is mediocre. We want to push this higher.

2. **MR-STFT distance is 1.42 (mean)** - this is fairly high, suggesting variations are deviating too much from the source in spectral terms. The p95 of 1.95 shows some outputs are very different.

3. **Token change rate is ~18.6%** - this seems reasonable but combined with the high MR-STFT, it suggests the tokens being changed are impactful ones causing large spectral shifts.

4. **Transient cross-correlation mean is 0.70 but p50 is 0.87** - there's a long tail of poor transient preservation. The attack_smear is close to 1.0 which is fine, but the mrstft_attack at 1.18 shows some attack region distortion.

5. **HF energy delta is -0.72 dB mean** - slight high-frequency energy loss on average, though p50 is near zero, suggesting some outliers are losing significant HF content.

6. **Spectral peak divergence is high (mean 20.8)** - the spectral peaks are shifting significantly.

Strategy:
- **Reduce mask_p_tail slightly (0.32 → 0.28)**: The tail region is where most masking occurs. Reducing this should produce less aggressive variations, lowering MR-STFT distance and improving acceptance rate.
- **Reduce temperature (0.9 → 0.75)**: Lower temperature will make the model's token predictions more conservative/likely, reducing spectral divergence and improving quality metrics.
- **Reduce top_p (0.95 → 0.9)**: Tightening the nucleus sampling will further constrain generation to higher-probability tokens.
- **Increase k_candidates (8 → 12)**: More candidates gives the selection process a better chance of finding high-quality variations that pass acceptance criteria.
- **Keep mask_p_attack at 0.08**: Attack preservation is reasonable (attack_smear ~0.96), no need to change this.

### Iteration 2

The acceptance rate has been slowly improving (0.508 → 0.517 → 0.533) and MR-STFT distance has been slowly decreasing (1.426 → 1.419 → 1.409), which are positive trends. However, the acceptance rate is still only 53%, meaning nearly half of generated variations are being rejected.

Key observations:
1. **hf_energy_delta_db is regressing** - mean is -0.675 dB, meaning we're losing high-frequency energy. This suggests the model is generating tokens that reduce HF content. Lowering temperature slightly could help produce more faithful reconstructions.
2. **token_change_rate is improving** at 0.16, which seems reasonable - not too much change, not too little.
3. **transient_xcorr** mean is 0.70 but p50 is 0.87, indicating some samples have very poor transient preservation (pulling the mean down). The attack_smear is near 1.0 which is good, so the attack mask protection is working.
4. **mrstft** at 1.41 is still quite high - we want variations that sound different but not drastically different.
5. **spectral_peak_divergence** is quite high at 20.6, suggesting spectral characteristics are shifting too much.

Strategy:
- **Increase k_candidates from 12 to 16**: More candidates means better filtering - we can pick higher quality outputs from a larger pool, which should improve acceptance rate.
- **Reduce mask_p_tail from 0.28 to 0.24**: The high spectral peak divergence and HF energy loss suggest we're masking too much of the tail, causing too-drastic changes. Reducing tail masking should help with HF preservation and reduce overall spectral divergence.
- **Reduce temperature from 0.75 to 0.65**: Lower temperature should produce more conservative token predictions, helping with the HF energy regression and reducing spectral peak divergence while keeping variations meaningful.

### Iteration 3

The acceptance rate has been steadily improving (0.508 → 0.517 → 0.533 → 0.533) and MR-STFT distance has been decreasing (1.426 → 1.419 → 1.409 → 1.402), both positive trends. However, the acceptance rate is still only ~53%, which is mediocre. Let me analyze the key issues:

1. **Transient preservation is inconsistent**: transient_xcorr mean is 0.706 but p50 is 0.871, indicating a subset of samples have very poor transient correlation, dragging down the mean and likely causing rejections.

2. **attack_smear is high (0.96)** which is good - attacks are mostly preserved. But mrstft_attack at 1.17 suggests some attack distortion remains.

3. **token_change_rate is low (0.137)** and improving - this means variations are relatively conservative, which is fine for quality but inter_var_mrstft_mean at 0.733 shows decent diversity.

4. **spectral_peak_divergence is quite high (20.55)** suggesting spectral artifacts in some outputs.

5. **hf_energy_delta_db mean is -0.70** meaning slight HF energy loss on average, though p50 is near zero (0.19), so a subset of samples lose significant HF energy.

The trends show token_change_rate and inter_var_mrstft_mean are improving, suggesting the current direction is working. I want to continue carefully:

- **Reduce mask_p_attack slightly (0.08 → 0.06)**: Further protect the attack region to improve transient_xcorr for the worst-case samples and reduce mrstft_attack. This should help the outlier samples that are being rejected.

- **Increase k_candidates (16 → 20)**: More candidates means better selection, which should help filter out the worst outputs with high spectral_peak_divergence and poor transient correlation.

- **Slight temperature decrease (0.65 → 0.55)**: The spectral_peak_divergence and HF energy loss outliers suggest some generations are too wild. Lowering temperature should reduce extreme outliers while maintaining the improving trends. This is a moderate step within limits.

- **Keep top_p and mask_p_tail unchanged**: mask_p_tail at 0.24 seems to be working well for the body of the sound, and top_p at 0.9 is reasonable.

### Iteration 4

The acceptance rate is ~53%, which is moderate but could be improved. Looking at the metrics:

1. **Acceptance rate (0.529)**: Roughly half of generated variations are being rejected. The median is 0.9375, meaning some sources produce mostly accepted variations while others produce mostly rejected ones - high variance.

2. **MR-STFT (1.40)**: Has been slowly decreasing over iterations (1.42 → 1.41 → 1.40 → 1.40), suggesting we're converging but still relatively high. Lower would indicate closer spectral fidelity.

3. **Token change rate (0.136)**: Relatively low, suggesting moderate variation levels. The mask_p values control this.

4. **Transient xcorr (0.702)**: Mean is okay but there's high variance (p50=0.866 vs mean=0.702), meaning some samples have poor transient preservation. The attack_smear is near 1.0 which is good.

5. **mrstft_attack (1.17)**: Attack spectral fidelity could be better. Reducing mask_p_attack slightly should help preserve transients better.

6. **spectral_peak_divergence (20.67)**: Quite high, suggesting frequency peaks are shifting too much in some cases.

Strategy: The system has been stable across iterations with gradual improvement. To push acceptance higher:
- **Reduce mask_p_attack from 0.08 to 0.06**: Better transient preservation should improve transient_xcorr and mrstft_attack, reducing rejections from poor transient quality.
- **Reduce mask_p_tail from 0.24 to 0.20**: Slightly less masking in the tail to reduce spectral divergence and improve overall MR-STFT, which should boost acceptance.
- **Increase k_candidates from 16 to 20**: More candidates gives the selection algorithm more options to pick better variations, directly improving acceptance and quality metrics.
- **Keep temperature and top_p the same**: Temperature at 0.65 and top_p at 0.9 seem reasonable. Don't want to change too many things at once.

### Iteration 5

Looking at the metrics, the system has been stable across iterations 2-5 with an acceptance rate stuck around 0.525-0.533 and MR-STFT around 1.40. Let me analyze the key issues:

1. **Acceptance rate (0.525)**: This is mediocre - nearly half of candidates are rejected. We want to improve this.

2. **transient_xcorr (mean=0.70, p50=0.86)**: The mean is dragged down significantly below the median, indicating some samples have very poor transient preservation. This is likely a major rejection driver.

3. **attack_smear (0.96)**: Close to 1.0 which is good, suggesting attacks are mostly preserved.

4. **token_change_rate (0.136)**: Relatively low, meaning variations aren't changing much. The mask_p values control how much gets masked.

5. **mrstft (1.40)**: This is in a reasonable range but could be lower for better quality.

6. **spectral_peak_divergence (mean=20.3, p95=38.4)**: Some samples have high spectral divergence, likely contributing to rejections.

7. **hf_energy_delta_db (mean=-0.71)**: Slight HF energy loss on average, but p50 is near 0 and p95 is +1.86, so there's a tail of samples losing HF energy.

The low transient_xcorr mean (0.70) with a much higher median (0.86) suggests some samples are badly damaging transients. Since attack_smear is fine, the issue might be in the tail region bleeding into the attack area, or temperature being too high for some tokens.

Strategy:
- **Reduce temperature slightly (0.65 → 0.55)**: Lower temperature should produce more conservative token choices, reducing the worst-case outliers that tank transient_xcorr and spectral_peak_divergence. This should improve acceptance rate.
- **Reduce top_p slightly (0.9 → 0.85)**: Tighter nucleus sampling will cut off unlikely tokens that produce bad artifacts.
- **Increase k_candidates (16 → 20)**: More candidates means better selection among generated variations, improving the quality of accepted outputs.
- **Keep mask_p_attack at 0.08**: Attack preservation seems okay.
- **Keep mask_p_tail at 0.24**: The variation amount seems reasonable given the token_change_rate.

The main theory is that the rejection-causing outliers come from high-variance token sampling, so tightening the sampling distribution while increasing candidate count should help.

## Listening Notes

_To be filled in after auditioning the listening pack._

### Machine Gun Test

Compare `machinegun_source.wav` (same hit repeated) vs `machinegun_variations.wav` (ML variations):

| Sample | Source MG | Variations MG | Notes |
|--------|----------|---------------|-------|
| pass-02_Track11_Kick_v025 | | | |
| pass-02_Track12_Kick_v025 | | | |
| pass-02_Track10_Kick_v025 | | | |
| pass-02_Track12_HiHatClosed_v076 | | | |
| pass-02_Track11_SnareRim_v025 | | | |
| pass-02_Track11_CrossStick_v051 | | | |

### Overall Impressions

- Identity preservation: 
- Variation quality: 
- Problem families: 
- Recommended next step: 
