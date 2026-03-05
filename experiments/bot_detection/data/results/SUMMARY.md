# Stage 3: Evaluation Results

## H1: Burstiness
- AUC-ROC: 0.490 [0.483, 0.496]
- AUC-PR: 0.252
- Mann-Whitney U: 138089817.0, p=0.001645

## H2: Engagement
- AUC-ROC: 0.500 [0.493, 0.506]
- AUC-PR: 0.253
- Mann-Whitney U: 140961788.0, p=0.945

## H3: Cross-repo fingerprinting
- AUC-ROC: 0.512 [0.506, 0.519]
- AUC-PR: 0.260
- Mann-Whitney U: 144455762.5, p=0.0002513

## H6: Interaction (burstiness x novelty)
- AUC-ROC: 0.493 [0.486, 0.499]
- AUC-PR: 0.253
- Mann-Whitney U: 138987461.0, p=0.02874

## H7: Burst content homogeneity
- AUC-ROC: 0.492 [0.486, 0.499]
- AUC-PR: 0.253
- Mann-Whitney U: 138907778.5, p=0.02305

## H1 Burstiness Sweep
- Configs tested: 15
- Significant after Holm-Bonferroni: 2
  - burst_count_24h_ge_1: adj_p=0.003651
  - burst_repos_24h_ge_1: adj_p=0.003651

## H3a vs H3b Comparison (DeLong)
- H3a AUC: 0.497, H3b AUC: 0.512
- z=-10.739, p=6.651e-27

## H4 Combined Model
- AUC-ROC: 0.518 [0.512, 0.524]
- LRT vs H1: chi2=154.57, df=8, p=2.177e-29
- LRT vs H2: chi2=175.93, df=9, p=3.589e-33
- LRT vs H3: chi2=116.26, df=9, p=7.776e-21

## H5 GE Complement (v1)
- GE-only AUC: 0.511
- GE+bot AUC: 0.519
- LRT: chi2=166.00, df=13, p=1.197e-28
- DeLong: z=3.980, p=6.905e-05

## H5 GE Complement (v2)
- GE-only AUC: 0.518
- GE+bot AUC: 0.521
- LRT: chi2=160.72, df=13, p=1.412e-27
- DeLong: z=2.849, p=0.00438
