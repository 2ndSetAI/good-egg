# Red Team Audit: Similarity Method Comparison

## Summary

Audit of `compare_similarity_methods.py` and its outputs. Found **9 issues** total: 0 critical, 2 major, 5 minor, 2 informational. The core finding -- that Jaccard retains the H4 signal while TF-IDF is borderline and MiniLM fails -- is robust, but the report omits multiple comparison correction that was specified in the original plan, and the decision criteria evaluation is incomplete.

## Issues

### Issue 1: No Multiple Comparison Correction Applied

- **Severity**: Major
- **Category**: Statistical Methodology / Spec Compliance
- **Description**: The original plan specified a Bonferroni correction threshold of 0.017 (0.05/3) for the three alternative methods. The script applies no correction at all, using a raw p < 0.05 threshold throughout. The report's decision criteria (lines 686-698) evaluate significance at uncorrected alpha = 0.05.
- **Evidence**: The `generate_report` function (line 698) checks `res["lrt_p_value"] < 0.05` with no correction. The plan says "TF-IDF retains H4 signal (LRT p < 0.017 Bonferroni)". There is no mention of Bonferroni in the script or report. Additionally, whether the denominator should be 3 (only new methods) or 4 (all methods including Gemini replication) is ambiguous.
- **Impact**: Does not change the Jaccard conclusion (p = 0.0002 passes any reasonable correction), but the TF-IDF categorization would change. Under Bonferroni with k=3 (threshold 0.017) or k=4 (threshold 0.0125), TF-IDF (p=0.064) remains non-significant. The report's narrative ("partially robust, 2/4 significant") is the same at either threshold, but the report should explicitly state the correction applied and why.
- **Recommendation**: Apply Bonferroni (or Holm-Bonferroni, which is already imported via `stats.py`) to the 3 non-Gemini LRT p-values. Report both raw and corrected p-values. The `holm_bonferroni()` function from stats.py is available but unused.

### Issue 2: Report Does Not Discuss AUC Inversion (All Methods Below 0.5)

- **Severity**: Major
- **Category**: Conclusion Validity
- **Description**: All four methods produce standalone AUC < 0.5 (Gemini: 0.416, TF-IDF: 0.442, MiniLM: 0.469, Jaccard: 0.441). An AUC below 0.5 means the similarity score is *negatively* associated with merge outcome -- higher similarity predicts non-merge. This counterintuitive result is not discussed anywhere in the report.
- **Evidence**: results.json shows all standalone_auc values below 0.5. The comparison_report.md mentions the AUC values in the table but never discusses the inversion or its implications for the H4 interpretation.
- **Impact**: The inverted AUC means that if similarity is "helping" prediction in the LRT, it does so by providing a *negative* signal (PRs more similar to the README are less likely to merge). This is an important interpretive point. It could indicate that boilerplate/template PRs match READMEs better but are lower quality, or that the similarity metric captures something other than "relevance." Without discussion, readers may assume similarity is positively associated with merge.
- **Recommendation**: Add a section discussing the AUC inversion, noting it is consistent across all methods (not a Gemini artifact). Discuss possible explanations and note that the LRT finding means similarity adds *information* (not that high similarity predicts merge).

### Issue 3: Pairwise DeLong Tests Lack Multiple Comparison Correction

- **Severity**: Minor
- **Category**: Statistical Methodology
- **Description**: Six pairwise DeLong tests are performed on standalone AUCs with no correction for multiple comparisons. The report marks "Gemini vs MiniLM" as significant (p = 0.0045) based on uncorrected p < 0.05.
- **Evidence**: Lines 844-855 compute 6 pairwise tests. `generate_report` line 669 marks significance at raw p < 0.05. With Bonferroni correction for 6 tests (threshold 0.0083), the "Gemini vs MiniLM" comparison (p = 0.0045) would still be significant, but this should be explicit.
- **Impact**: Low in this case since the one significant result survives correction, but methodologically the correction should be stated.
- **Recommendation**: Apply Holm-Bonferroni or note the Bonferroni-corrected threshold (0.05/6 = 0.0083) in the report.

### Issue 4: Title Fallback Source Differs from Stage5

- **Severity**: Minor
- **Category**: Data Integrity
- **Description**: When a PR body is empty and the script falls back to the title, it sources the title from the JSONL file (`pr_title_map`), while stage5 sources it from the dataframe row (`row.get("title", "")`). These likely contain the same data since both originate from the same PR metadata, but the sources differ.
- **Evidence**: Stage5 line 248-249: `text = str(row.get("title", ""))`. Compare script lines 104, 125, 128-129: title comes from `pr_title_map.get(pr_key, "")` which is loaded from the JSONL at line 112.
- **Impact**: Minimal. If any PR has a title in the parquet but not in the JSONL (or vice versa), the text pair could differ, producing a slightly different similarity score. The sanity check (which validates Gemini cache matches) should catch any resulting discrepancy, and it passes.
- **Recommendation**: Document this divergence or align the title source with stage5. Since the sanity check passes, this is unlikely to affect results.

### Issue 5: Fragile Variable Scoping in `generate_figure`

- **Severity**: Minor
- **Category**: Code Quality
- **Description**: The `merged_colors` variable (line 529) and `legend_elements` variable (lines 544-552) are defined inside the `if "TF-IDF" in method_sims and "Gemini" in method_sims:` block but used in the subsequent `if "MiniLM" in method_sims and "Gemini" in method_sims:` block (lines 559, 567). If TF-IDF were absent but MiniLM present, this would raise a `NameError`.
- **Evidence**: Lines 529 and 559 both reference `merged_colors`. Lines 544 and 567 both reference `legend_elements`. These are defined conditionally.
- **Impact**: No practical impact since TF-IDF has no optional dependencies and is always computed. But the code would break if TF-IDF computation were ever made optional.
- **Recommendation**: Move `merged_colors` definition above the conditional blocks, or pass it as a parameter.

### Issue 6: Wasteful Double Computation of Marginal Improvement

- **Severity**: Minor
- **Category**: Code Quality
- **Description**: `analyze_method` (line 461) computes `run_marginal_improvement` with zero-filled NaN values for merge_rate and age. Then at lines 830-839, the marginal is recomputed with properly masked data, overwriting the first result. The first computation is wasted.
- **Evidence**: Line 819-827 calls `analyze_method` with `merge_rate_clean` and `age_clean` (zero-filled). Lines 830-839 immediately overwrite `result["marginal"]` with the correctly filtered version.
- **Impact**: No effect on results (the incorrect first computation is always overwritten), but wastes compute and is confusing to read.
- **Recommendation**: Either skip the marginal in `analyze_method` when merge_rate/age will be zero-filled, or pass the `has_extras` mask to `analyze_method` to avoid the redundant computation.

### Issue 7: The 48-Row Difference (n_with_extras=1,245 vs n_pairs=1,293) Is Not Explained in the Report

- **Severity**: Minor
- **Category**: Conclusion Validity
- **Description**: The results.json records `n_pairs: 1293` and `n_with_extras: 1245`, meaning 48 rows lack valid `author_merge_rate` or `log_account_age_days`. The marginal improvement analysis uses the 1,245-row subset while the LRT and standalone AUC use all 1,293. This difference is not explained in the report.
- **Evidence**: results.json shows both values. The comparison_report.md says "same 1,293 PR body / repo README text pairs" but the marginal tables silently use 1,245.
- **Impact**: Minor. The marginal analysis operates on a slightly different subset than the LRT, which is methodologically appropriate (only use rows with valid features), but the difference should be documented.
- **Recommendation**: Note in the report that the marginal improvement analysis uses n=1,245 rows with valid merge_rate and account_age data.

### Issue 8: MiniLM 128-Token Truncation Deserves Deeper Analysis

- **Severity**: Informational
- **Category**: Conclusion Validity
- **Description**: MiniLM uses 128-token truncation (line 264), which is severely limiting for README texts truncated to 4,000 characters (roughly 600-1000 tokens). The report mentions this as a limitation but does not quantify the impact. Most of the README content is simply discarded.
- **Evidence**: Line 264: `tokenizer.enable_truncation(max_length=128)`. READMEs are up to 4,000 characters. With average word length ~5 characters, that's ~800 words. MiniLM's tokenizer likely produces more tokens than words, so 128 tokens represents perhaps 10-15% of the README. The Pearson r with Gemini is only 0.438 (vs 0.586 for TF-IDF), consistent with severe information loss.
- **Impact**: MiniLM's failure (LRT p=0.381) is likely explained primarily by truncation rather than model quality. The standard MiniLM max sequence length is 256 tokens; 128 is even more aggressive. This limitation means MiniLM cannot be fairly evaluated under these conditions.
- **Recommendation**: Note in the report that MiniLM's 128-token truncation discards ~85-90% of README content, making its failure expected. Consider a follow-up test with 256 or 512 tokens (with model fine-tuning or chunking) before concluding MiniLM is inadequate.

### Issue 9: Selection Bias in the 26% Subsample Is Acknowledged But Not Quantified

- **Severity**: Informational
- **Category**: Conclusion Validity
- **Description**: The analysis operates on 1,293 out of 4,977 PRs (26%) -- only those with both a non-empty PR body/title AND a repo README. This selects for (a) more active/documented repos and (b) PRs with descriptive bodies, which may correlate with merge outcome.
- **Evidence**: `n_pairs=1293` out of 4,977 total PRs (from pilot_report.md). The report's Known Limitations section does not mention the selection percentage or its implications.
- **Impact**: The H4 finding applies only to the 26% of PRs with text pairs. The selected subset may have different merge rates, author profiles, or repo characteristics than the full dataset. This affects generalizability but not internal validity.
- **Recommendation**: Report the selection percentage and compare merge rates between the 1,293-row subset and the full 4,977-row dataset. If they differ substantially, note the generalizability limitation.

## Verified Correct

1. **Gemini LRT matches stage6**: The comparison script's Gemini LRT statistic (20.818, p=5.05e-6) exactly matches `statistical_tests.json`'s `H4_embedding_lrt.lr_statistic` (20.818, p=5.05e-6). The same 1,293 rows are used (both filter on `embedding_similarity.notna()`).

2. **H4 LRT base model matches stage6**: Both stage6 and the comparison script use `LR(GE_score)` as the base model and `LR(GE_score, similarity)` as the full model, with `penalty=None`, `max_iter=1000`, `random_state=42`. The `log_loss_manual` implementation is identical.

3. **`log_loss_manual` is correctly implemented**: The function (line 62-68) matches stage6's version (line 894-902) exactly. Both clip predictions, compute mean binary cross-entropy, and negate it. The log-likelihood is computed as `-log_loss_manual(...) * n` which gives the total log-likelihood.

4. **`likelihood_ratio_test` from stats.py used correctly**: The LRT statistic is `-2 * (ll_null - ll_alt)` which is correct for nested logistic regression models. The chi-squared test with df=1 is appropriate for one additional parameter.

5. **TF-IDF parameters match spec**: `max_features=10000`, `stop_words="english"`, `sublinear_tf=True`, `min_df=2`, `ngram_range=(1, 2)` at lines 217-222.

6. **MiniLM implementation is correct**: ONNX inference with proper mean-pooling and L2 normalization (lines 291-304). The attention mask is correctly applied during pooling.

7. **Jaccard implementation is correct**: Lowercased word sets with standard Jaccard formula (lines 325-332). Handles empty union edge case.

8. **Text truncation matches spec**: READMEs truncated to 4,000 chars (line 100), PR bodies to 2,000 chars (line 131), with title fallback if body empty (lines 128-129).

9. **Sanity check validates what it claims**: The check (lines 145-204) loads Gemini embeddings from cache, recomputes cosine similarity, and verifies within 1e-6 of the parquet value. This confirms the text pair reconstruction matches what stage5 used.

10. **DeLong test implementation is correct**: The `delong_auc_test` in stats.py correctly computes placement values and the covariance matrix. Pairwise DeLong comparisons between methods are correctly applied.

11. **Marginal improvement uses correct feature set**: The rerun at lines 830-839 uses properly filtered data (rows with valid merge_rate and account_age), matching the H3 pattern in stage6.

12. **LogisticRegression uses `penalty=None` consistently**: All 6 LogisticRegression instances in the comparison script use `penalty=None`, matching stage6's approach and ensuring the LRT chi-squared assumption holds.

13. **Correlation tests are appropriate**: Pearson r measures linear correlation between similarity scores; Spearman r measures monotonic correlation. Both are standard for comparing similarity measures.

14. **Output format matches spec**: results.json, comparison_report.md, and similarity_comparison.png (4 panels: forest, KDE, TF-IDF scatter, MiniLM scatter) are all generated as specified.

## Overall Assessment

The Jaccard recommendation is **trustworthy**. Jaccard similarity retains the H4 signal (LRT p=0.0002, which survives any reasonable multiple comparison correction) with Pearson r=0.50 against Gemini. No results need to be re-run.

The two major issues (missing Bonferroni correction and undiscussed AUC inversion) affect the report's completeness and rigor but do not change the substantive conclusion. The Jaccard LRT p-value is so small (0.0002) that it passes under Bonferroni with k=3 (threshold 0.017), k=4 (threshold 0.0125), or even k=10. The AUC inversion is consistent across all methods and is an interpretive issue, not a computational one.

The comparison correctly demonstrates that the H4 finding is not an artifact of Gemini's specific embedding model, since a trivial bag-of-words method (Jaccard) captures the same predictive signal. This strengthens the case that there is real semantic information in the PR-README text pair, while also raising the question of why simpler methods suffice (suggesting the signal may be more lexical than semantic).
