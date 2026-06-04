# Submission Readiness Gaps

Items requiring **human** completion before submission. None affect the
scientific content, which is locked and audited
(`FINAL_MANUSCRIPT_CLAIM_TO_RESULT_AUDIT.md`). These are metadata/formatting
gaps only; they were intentionally left as placeholders rather than invented.

## Author / affiliation metadata

- [ ] Confirm author order and equal-contribution notation (currently
      **Arush Ravipati**$^{\ast}$, **Abhijay Gangarapu**$^{\ast}$, equal).
- [ ] Real affiliation(s) — currently "Independent Researcher" placeholder.
- [ ] Corresponding-author designation, email, and ORCID iDs (none stated).
- [ ] Postal/institutional addresses if the target venue requires them.

## Venue / formatting

- [ ] Choose target venue and apply its class/template. Current build uses the
      generic `article` class (11pt) compiled with **tectonic**; the repo has no
      `pdflatex`/`latexmk` (`make paper` assumes `pdflatex`). Update the build
      command for the chosen venue.
- [ ] Convert the manual `thebibliography` in `references.tex` to the venue's
      required citation style (e.g., BibTeX/`.bbl`) if mandated.
- [ ] Keyword list, running title, and abstract word-limit check.
- [ ] Resolve cosmetic overfull-hbox warnings (long URL in the COG-BCI reference;
      a few wide table/paragraph lines) if the venue enforces strict typesetting.

## Funding / ethics / declarations

- [ ] Funding statement (none claimed; add or state "no funding").
- [ ] Data-availability statement may need venue-specific accession links/DOIs
      (PhysioNet EEGMAT, IEEE DataPort STEW, COG-BCI Zenodo are cited).
- [ ] Confirm the secondary-analysis ethics statement satisfies venue policy
      (no new human-subjects data were collected).
- [ ] Conflict-of-interest and author-contribution statements are present but
      should be verified against venue wording.

## Code / reproducibility

- [ ] Public code/data repository link or archived DOI (e.g., Zenodo) for the
      analysis code, frozen configs, and committed result summaries.
- [ ] Note: `scripts/make_negative_manuscript_figures.py` requires `matplotlib`
      + `numpy`; document the environment if reproducibility is reviewed.
- [ ] Optional: regenerate the MAT no-gamma feature parquet from
      provenance-verified raw data (known reproducibility gap noted in
      `docs/EEGARUSH_CHECKPOINT.md`) if a clean-clone full-test pass is required.

## Explicitly NOT gaps (do not "fix" by adding claims)

- The negative result is the finding; do not add a positive/transfer claim,
  a rescue analysis, or a new dataset to "strengthen" the paper.
- Do not add clinical, mechanistic, anatomical, PAC/gamma, source-localization,
  invariant-axis, or DS007262 content. These are forbidden per the claim ledger.
- Do not estimate or assert publication-acceptance likelihood.
