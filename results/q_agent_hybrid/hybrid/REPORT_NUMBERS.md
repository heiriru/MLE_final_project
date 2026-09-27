# Hybrid experiments: report numbers
Score per round = coins + 5 * kills. Mean +- standard error over seat blocks; resolved if |mean| > 2 SE.

## H1/H2 correctness gate: frozen adapter determinism and zero-residual parity

- 25 fixed-seed classic rounds (seed 1201), 6547 adapter decisions
- adapter run A == run B: True
- zero-residual hybrid matches adapter: 6547/6547
- parity exact: True

## H0 noise floor: q-agent against an identical copy of itself (4 seats x 200 rounds, seed 4801)

| comparison | q-agent v1 | q-agent v1 (identical copy) | difference | status |
|---|---|---|---|---|
| identical tables | 3.195 +- 0.117 | 3.440 +- 0.053 | -0.245 +- 0.142 | unresolved |

## H1 adapter ablation (25 rounds, seed 909, adapter + 3 rule_based)

| adapter mode | score/round | coins | kills | suicides |
|---|---|---|---|---|
| full | 3.320 | 1.920 | 0.280 | 0.160 |
| no_q_table | 3.320 | 2.120 | 0.240 | 0.480 |
| unseen_wait | 4.240 | 2.040 | 0.440 | 0.160 |

## H3-H8 early screens: hybrid vs q-agent, four 25-round seat blocks, SE over blocks

| stage | run | hybrid | q-agent | hybrid - q-agent | status |
|---|---|---|---|---|---|
| H3 | first residual DDQN (seed 1401) | 3.160 +- 0.175 | 3.480 +- 0.156 | -0.320 +- 0.327 | unresolved |
| H4-1 | 1-step, seed 1501 | 2.880 +- 0.192 | 3.030 +- 0.286 | -0.150 +- 0.447 | unresolved |
| H4-1 | 1-step, seed 1502 | 3.360 +- 0.313 | 3.360 +- 0.256 | +0.000 +- 0.091 | unresolved |
| H4-1 | 1-step, seed 1503 | 3.580 +- 0.334 | 3.670 +- 0.310 | -0.090 +- 0.563 | unresolved |
| H4-3 | 3-step, seed 1501 | 3.550 +- 0.307 | 3.380 +- 0.295 | +0.170 +- 0.206 | unresolved |
| H4-3 | 3-step, seed 1502 | 3.500 +- 0.158 | 2.780 +- 0.200 | +0.720 +- 0.241 | resolved |
| H4-3 | 3-step, seed 1503 | 2.780 +- 0.165 | 2.960 +- 0.110 | -0.180 +- 0.143 | unresolved |
| H5-C | conservative training, n=3 | 3.470 +- 0.428 | 3.200 +- 0.313 | +0.270 +- 0.704 | unresolved |
| H5-S | potential shaping, n=3 | 3.580 +- 0.150 | 3.760 +- 0.326 | -0.180 +- 0.336 | unresolved |
| H6 | replay/training repair | 3.630 +- 0.206 | 3.270 +- 0.224 | +0.360 +- 0.307 | unresolved |
| H7 | six-input residual | 3.440 +- 0.273 | 3.210 +- 0.381 | +0.230 +- 0.365 | unresolved |
| H8-B | antithetic ES, best vector | 3.510 +- 0.294 | 3.290 +- 0.281 | +0.220 +- 0.471 | unresolved |
| H8-M | antithetic ES, mean vector | 3.600 +- 0.352 | 3.010 +- 0.206 | +0.590 +- 0.538 | unresolved |

## H4 pooled over the three training seeds (12 seat blocks per return length)

| return | hybrid | q-agent | hybrid - q-agent | status |
|---|---|---|---|---|
| 1-step | 3.273 +- 0.174 | 3.353 +- 0.168 | -0.080 +- 0.219 | unresolved |
| 3-step | 3.277 +- 0.157 | 3.040 +- 0.136 | +0.237 +- 0.153 | unresolved |

3-step minus 1-step, paired by training seed (n=3 seeds): +0.003 +- 0.430 (unresolved); per seed +0.670, +0.140, -0.800

## H8 direct score search: optimisation traces

- H8 hill climbing: 10 iterations, 4 candidates accepted
- H8 Adam-ES: 1000 epochs; `score_per_round` first 100 epochs 3.322 +- 0.129, last 100 3.428 +- 0.117
- H8 antithetic Adam-ES: 1000 epochs; `plus_score_per_round` first 100 epochs 3.560 +- 0.152, last 100 3.134 +- 0.135

## H9 calibration sweep of the H7 checkpoint (200 rounds, seed 2201, roster H9, H7, q-agent, rule)

| limit L | margin m | budget 2L-m | H9 score | H7 (L=.25,m=.25) | q-agent |
|---|---|---|---|---|---|
| 0.15 | 0.10 | 0.20 | 3.055 | 3.235 | 3.275 |
| 0.15 | 0.25 | 0.05 | 3.335 | 3.125 | 3.220 |
| 0.15 | 0.40 | -0.10 | 3.150 | 3.335 | 3.370 |
| 0.25 | 0.10 | 0.40 | 3.425 | 3.385 | 3.130 |
| 0.25 | 0.25 | 0.25 | 3.615 | 3.340 | 3.375 |
| 0.25 | 0.40 | 0.10 | 3.055 | 3.430 | 3.495 |
| 0.35 | 0.10 | 0.60 | 3.040 | 3.290 | 3.265 |
| 0.35 | 0.25 | 0.45 | 3.380 | 3.105 | 3.650 |
| 0.35 | 0.40 | 0.30 | 2.895 | 3.225 | 3.350 |

## H9-H12 common ladder: {stage, rule, rule, q-agent}, classic, seed 4801, 4 seats x 200 rounds

| stage | change | stage score | rule_based | q-agent | stage - q-agent | status | stage - rule | status |
|---|---|---|---|---|---|---|---|---|
| H9 | L=.25, m=.25, 6 inputs (H7 checkpoint) | 3.303 +- 0.128 | 3.125 +- 0.048 | 3.386 +- 0.068 | -0.084 +- 0.185 | unresolved | +0.177 +- 0.167 | unresolved |
| H10 | add scalar value head | 3.225 +- 0.178 | 3.144 +- 0.025 | 3.297 +- 0.113 | -0.073 +- 0.265 | unresolved | +0.081 +- 0.166 | unresolved |
| H11 | add eight observations | 3.290 +- 0.172 | 3.131 +- 0.041 | 3.329 +- 0.086 | -0.039 +- 0.153 | unresolved | +0.159 +- 0.205 | unresolved |
| H12 | L=1.00, m=.05, anchor .01 | 3.612 +- 0.083 | 2.904 +- 0.016 | 3.624 +- 0.097 | -0.011 +- 0.176 | unresolved | +0.708 +- 0.073 | resolved |

Consecutive stages (unpaired, different games):

- H10 - H9: -0.078 +- 0.220 (unresolved)
- H11 - H10: +0.065 +- 0.248 (unresolved)
- H12 - H11: +0.322 +- 0.191 (unresolved)

Behaviour per round (coins, kills, suicides):

| stage | coins | kills | suicides |
|---|---|---|---|
| H9 | 1.809 +- 0.052 | 0.299 +- 0.019 | 0.305 +- 0.012 |
| H10 | 1.806 +- 0.063 | 0.284 +- 0.026 | 0.294 +- 0.014 |
| H11 | 1.828 +- 0.043 | 0.292 +- 0.028 | 0.314 +- 0.011 |
| H12 | 2.175 +- 0.067 | 0.287 +- 0.010 | 0.255 +- 0.004 |

## Direct head-to-head blocks (4 seats x 200 rounds, seed 4801)

| comparison | first | second | first - second | status |
|---|---|---|---|---|
| H10 vs H9 (H10, H9, q-agent, rule) | 3.229 +- 0.076 | 3.380 +- 0.125 | -0.151 +- 0.159 | unresolved |
| H11 vs H10 (H11, H10, q-agent, rule) | 3.270 +- 0.115 | 3.114 +- 0.036 | +0.156 +- 0.145 | unresolved |
| H12 vs H9 (H12, H9, q-agent, rule) | 3.679 +- 0.104 | 3.243 +- 0.089 | +0.436 +- 0.181 | resolved |
| H12 vs q-agent (H12, H9, q-agent, rule) | 3.679 +- 0.104 | 3.188 +- 0.139 | +0.491 +- 0.122 | resolved |
| H12 vs rule (H12, H9, q-agent, rule) | 3.679 +- 0.104 | 3.471 +- 0.049 | +0.207 +- 0.090 | resolved |

## Powered H12 vs q-agent: {H12, rule, rule, q-agent}, 8 seed bases x 4 seats x 200 rounds

8 complete seed bases.

| base | H12 | q-agent | H12 - q-agent |
|---|---|---|---|
| 6001 | 3.766 | 3.482 | +0.284 |
| 6101 | 3.564 | 3.489 | +0.075 |
| 6201 | 3.726 | 3.583 | +0.144 |
| 6301 | 3.740 | 3.631 | +0.109 |
| 6401 | 3.759 | 3.652 | +0.106 |
| 6501 | 3.609 | 3.476 | +0.133 |
| 6601 | 3.516 | 3.456 | +0.060 |
| 6701 | 3.366 | 3.516 | -0.150 |

- seed-base estimator: H12 3.631 +- 0.051, q-agent 3.536 +- 0.027, difference +0.095 +- 0.043 (resolved)
- Student-t 95% interval over the 8 base differences: [-0.006, +0.196]
- round-paired estimator over 6400 rounds: +0.095 +- 0.057 (unresolved); per-round sd 4.56
- rounds needed to resolve an effect of this size: ~9208

| agent | coins | kills | suicides |
|---|---|---|---|
| hybrid_h12 | 2.218 +- 0.022 | 0.283 +- 0.007 | 0.263 +- 0.005 |
| q_agent_v1 | 1.801 +- 0.017 | 0.347 +- 0.006 | 0.266 +- 0.006 |

## Held-out roster: agent + 3 rule_based, seed 5801, 4 seats x 200 rounds (same maps for H12 and q-agent)

| agent | score | coins | kills | suicides |
|---|---|---|---|---|
| H12 | 3.911 +- 0.073 | 2.136 +- 0.041 | 0.355 +- 0.020 | 0.226 +- 0.020 |
| q-agent | 3.755 +- 0.151 | 1.974 +- 0.068 | 0.356 +- 0.019 | 0.166 +- 0.016 |

seat-paired H12 - q-agent: +0.156 +- 0.163 (unresolved)

## H12 training-seed replication (seeds 1502, 1503), eval seed 8201, 4 seats x 100 rounds

| train seed | field | H12 | q-agent | H12 - q-agent | status |
|---|---|---|---|---|---|
| 1502 | H12, q-agent, rule, rule | 3.485 +- 0.078 | 3.147 +- 0.180 | +0.337 +- 0.160 | resolved |
| 1502 | H12, rule, rule, rule | 3.742 +- 0.069 | 3.595 +- 0.112 | +0.147 +- 0.178 | unresolved |
| 1503 | H12, q-agent, rule, rule | 3.468 +- 0.183 | 3.350 +- 0.127 | +0.118 +- 0.263 | unresolved |
| 1503 | H12, rule, rule, rule | 3.527 +- 0.214 | 3.595 +- 0.112 | -0.067 +- 0.102 | unresolved |

Behaviour in the held-out field (coins, kills, suicides per round):

- seed 1502: 2.292, 0.290, 0.188
- seed 1503: 2.215, 0.263, 0.250
- q-agent control: 1.995, 0.320, 0.215

## Training diagnostics (final row of metrics.jsonl)

| run | rounds | interactions | updates | final epsilon | mean loss | mean round reward |
|---|---|---|---|---|---|---|
| H3_residual_ddqn_seed1401 | 2000 | 364348 | 181675 | 0.050 | 0.678 | -4.95 |
| H5_conservative_seed1501 | 2000 | 350270 | 174661 | 0.050 | 1.059 | -5.05 |
| H5_shaping_seed1501 | 2000 | 365906 | 182502 | 0.050 | 1.076 | -4.86 |
| H6_replay_repair_seed1501 | 2000 | 550512 | 135122 | 0.010 | 0.909 | -1.17 |
| H7_six_inputs_seed1501 | 2000 | 558467 | 137051 | 0.010 | 0.916 | -0.63 |
| H10_dueling_seed1501 | 2000 | 551820 | 135428 | 0.010 | 0.091 | -0.93 |
| H11_features_seed1501 | 2000 | 555789 | 136417 | 0.010 | 0.094 | -2.15 |
| H12_authority_seed1501 | 2000 | 546820 | 134235 | 0.010 | 0.085 | -0.95 |
| H12_replication_seed1502 | 2000 | 564893 | 138760 | 0.010 | 0.080 | -1.21 |
| H12_replication_seed1503 | 2000 | 541165 | 132803 | 0.010 | 0.091 | -0.74 |
| H4_nstep_equal_budget/n1_seed1501 | 2000 | 371513 | 185257 | 0.050 | 0.690 | -3.99 |
| H4_nstep_equal_budget/n1_seed1502 | 2000 | 355193 | 177097 | 0.050 | 0.698 | -3.94 |
| H4_nstep_equal_budget/n1_seed1503 | 2000 | 349112 | 174057 | 0.050 | 0.688 | -4.92 |
| H4_nstep_equal_budget/n3_seed1501 | 2000 | 352359 | 175698 | 0.050 | 1.038 | -4.53 |
| H4_nstep_equal_budget/n3_seed1502 | 2000 | 362314 | 180648 | 0.050 | 1.080 | -4.53 |
| H4_nstep_equal_budget/n3_seed1503 | 2000 | 354721 | 176864 | 0.050 | 1.095 | -6.06 |

## Residual override rate of the trained checkpoints (50 rounds, seed 4801)

hybrid_h3_h8 = H9 (H7 checkpoint, limit 0.25, margin 0.25)

| agent | decisions | override rate | mean largest correction |
|---|---|---|---|
| hybrid_h10 | 15277 | 0.0001 | 0.199 |
| hybrid_h11 | 15717 | 0.0003 | 0.203 |
| hybrid_h12 | 15551 | 0.1493 | 0.838 |
| hybrid_h3_h8 | 16046 | 0.0000 | 0.250 |
