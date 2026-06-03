# Regression Check Report

Scope: read-only audit of the current repository state. No LaTeX, Python, or analysis scripts were executed; this report is based on file inspection and existing artifacts only.

## Summary Status

| Item | Status | Evidence |
|---|---|---|
| `arjun` to `arush` replacement in `main.tex` | PASS | No `arjun` or `Arjun` instances were found in `paper/tex/main.tex`; author block contains `Arush Ravipati`. |
| Author formatting | PASS, compile not verified | `paper/tex/main.tex` lists `Abhijay Gangarapu$^{\ast}$` and `Arush Ravipati$^{\ast}$` and includes an equal-contribution footnote after `\maketitle`. A second Author Contributions statement also says both contributed equally. |
| Equal-collaborator first-page statement | PASS, compile not verified | The footnote text is present: `$^{\ast}$These authors contributed equally to this work.` |
| PAC references in manuscript body | PASS | No body-text matches for PAC, phase-amplitude coupling, theta-gamma, or cross-frequency coupling were found in `main.tex`. |
| Lingering PAC bibliography entries | FAIL | `paper/tex/references.tex` still includes theta-gamma/PAC bibliography items. Because `main.tex` directly inputs this bibliography file, these may appear in the rendered bibliography even if uncited. |
| Lingering PAC scripts/artifacts | PRESENT | PAC-related scripts and figures remain, including `scripts/theta_gamma_coupling.py`, `scripts/theta_gamma_coupling_v2.py`, `scripts/plot_pac_results.py`, `paper/figures/figure_pac_results.png`, and `paper/figures/comodulogram_fz.png`. This is not automatically wrong, but it is stale relative to the current manuscript. |
| Inverted exclamation math bug (`¡`) | PASS | No `¡` characters were found in inspected TeX, Markdown, or Python files. |
| Plain-text math comparisons | FAIL | `main.tex` still contains plain-text comparison/math strings such as `p<0.001`, `empirical p<0.001`, and `AUC <= 0.5` outside math mode. The issue is formatting, not the old inverted-exclamation bug. |
| Source-localization placeholders | FAIL | `main.tex` still contains TODO source-localization methods/results sections and later states source localization was not performed. |
| Repository URL metadata | FAIL | `main.tex` reports `https://github.com/agangarapu/eeg-workload-replication`, not the current repository `https://github.com/CoderAwesomeAbhi/EEGARUSH`. |
| Hyperparameter-search claim | FAIL | The manuscript's claim about inner-fold SVM hyperparameter search and default settings in 78 percent of folds was not matched to executable code or a result ledger. |
| Channel/common-feature consistency | FAIL | The manuscript references both an 8-channel intersection and a 10-channel common feature set. The code has multiple channel-intersection definitions, so the paper needs one auditable definition per analysis. |
| SNWA selected-feature description | FAIL | The manuscript's listed eight SNWA features do not match `outputs_journal_upgrade/tables/table_snwa_feature_stability.csv`. |
| Feature-family stability percentages | FAIL | Current family-stability artifact counts do not directly support the manuscript's reported percentages. |
| Connectivity feature count | FAIL | Manuscript text reports 171 pairwise correlation features, while `outputs_phd_revision/tables/sr10_connectivity_equiv.csv` reports 215 connectivity features. |
| External SNWA provenance | FAIL | Static DS007262 SNWA predictions and t-tests exist, but no clear executable generator was found for `external_validation_ds007262/ds007262_low_high_predictions.csv`. |
| External-validation artifact consistency | FAIL | `outputs_journal_upgrade/tables/table_external_validation_metrics.csv` says external validation was `not_run`, while other folders contain DS007262 external prediction/t-test artifacts. |
| Hard-coded local path in exploratory script | FAIL | `scripts/option_a_individual_variability.py` contains a hard-coded Windows path, which makes that analysis non-portable without manual edits. |
| Log-transform sensitivity implementation | FAIL | `scripts/finish_everything.py` fits a new scaler on the test split in the log-transform check instead of applying the training scaler, weakening that sensitivity result. |
| LaTeX compile status | NOT RUN | Compilation was intentionally not run during this read-only audit. |

## Bottom Line

The authorship and `arush` metadata fixes are present in `main.tex`, and the old `¡` symbol regression does not appear to remain. The biggest remaining risks are scientific provenance and overclaiming: invariant-axis language, biological source claims, stale PAC bibliography/artifacts, unsupported hyperparameter/search statements, mismatched SNWA feature descriptions, and incomplete external SNWA generation traceability.

