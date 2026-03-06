# Stage 3: Evaluation Results

## H1: Burstiness
- AUC-ROC: 0.483 [0.480, 0.486]
- AUC-PR: 0.263
- Mann-Whitney U: 3820139298.0, p=4.55e-32

## H2: Engagement
- AUC-ROC: 0.481 [0.478, 0.484]
- AUC-PR: 0.263
- Mann-Whitney U: 3804590998.0, p=1.434e-39

## H3: Cross-repo fingerprinting
- AUC-ROC: 0.503 [0.500, 0.506]
- AUC-PR: 0.271
- Mann-Whitney U: 3977369222.0, p=0.03234

## H6: Interaction (burstiness x novelty)
- AUC-ROC: 0.480 [0.477, 0.483]
- AUC-PR: 0.265
- Mann-Whitney U: 3796364322.0, p=4.7e-44

## H7: Burst content homogeneity
- AUC-ROC: 0.479 [0.477, 0.482]
- AUC-PR: 0.262
- Mann-Whitney U: 3789499922.5, p=7.93e-48

## H1 Burstiness Sweep
- Configs tested: 19
- Significant after Holm-Bonferroni: 7
  - burst_repos_24h_ge_2: adj_p=0.0001831
  - burst_count_24h_ge_1: adj_p=0.000247
  - burst_repos_24h_ge_1: adj_p=0.000247
  - burst_repos_24h_ge_3: adj_p=0.0002843
  - burst_count_24h_ge_7: adj_p=0.0003293

## H3a vs H3b Comparison (DeLong)
- H3a AUC: 0.493, H3b AUC: 0.503
- z=-20.279, p=1.951e-91

## H4 Combined Model
- AUC-ROC: 0.501 [0.499, 0.504]
- LRT vs H1: chi2=329.79, df=8, p=1.859e-66
- LRT vs H2: chi2=347.74, df=9, p=1.872e-69
- LRT vs H3: chi2=34.30, df=9, p=7.906e-05

## H5 GE Complement (v1)
- GE-only AUC: 0.497
- GE+bot AUC: 0.502
- LRT: chi2=207.62, df=13, p=3.712e-37
- DeLong: z=7.862, p=3.792e-15

## H5 GE Complement (v2)
- GE-only AUC: 0.521
- GE+bot AUC: 0.520
- LRT: chi2=180.42, df=13, p=1.394e-31
- DeLong: z=-2.519, p=0.01177
