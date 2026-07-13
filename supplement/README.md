# Supplementary Information — *Beyond Arbitrariness: Detecting Geographic Structure and the Limits of Current Linguistic Databases*

This directory is a self-contained **audit and reproducibility layer** for the main
manuscript. It introduces no new estimates, models, or interpretations. Every
publication-facing quantity is traceable:

> manuscript → SI table/figure → Result CSV → the exact notebook/Python code that generated it.

## How the SI is generated

All tables and figures are produced **programmatically from the Result CSV files** in
`../results/` (the source of truth). No number is transcribed from the manuscript,
the notebook display output, or a PDF. Rule/prior text is taken verbatim from the
notebook code cells (6, 11, 70).

Pipeline (run in order, from `supplement/scripts/`):

```bash
cd supplement/scripts
python3 inventory_notebook.py               # -> generated/notebook_inventory.csv
python3 build_manifest.py                    # -> manifest.json, generated/manuscript_contract.json (+schema)
python3 build_source_ledger.py               # -> generated/source_ledger.csv, generated/claim_to_evidence_map.csv
python3 build_si_tables.py                   # -> tables/tableS1..S18.tex
python3 build_si_figures.py                  # -> figures/figureS1..S3 (+ *_source.csv)
python3 validate_manuscript_consistency.py   # -> generated/validation_report.md, generated/manuscript_consistency_report.md
bash    compile_supplement.sh                # -> supplement.pdf
```

Dependencies (CPU-only): `python3`, `nbformat`, `matplotlib`, a LaTeX toolchain
(`latexmk` + `elsarticle`, `booktabs`, `graphicx`, `lmodern`, `microtype`).

## Authoritative files

| Purpose | File | Authoritative for |
|---|---|---|
| Numerical results | `../results/*.csv` | all coefficients, CIs, counts, diagnostics, PPC, robustness |
| Provenance/versions | `../results/reproducibility_metadata.json` | seed, SHAs, backend, device |
| Rules/priors/models | `../Beyond_Arbitrariness_BetaBinomial_Diagnostics_final.ipynb` (cells 6, 11, 70, …) | preprocessing, priors, likelihoods, sampler settings |
| Derived manifest | `manifest.json` | consolidated publication-facing values (posterior medians) |

Result CSVs are **never** overwritten by these scripts; the SI only reads them.

## Why the H100 models are NOT rerun

The reported models are NumPyro/NUTS Beta-Binomial fits run on an H100/A100 GPU.
Their final outputs already exist as committed Result CSVs (and named NetCDF
artifacts). Rerunning them is expensive, non-deterministic across hardware, and
unnecessary: the SI audits the stored outputs. Only lightweight CPU code runs here
(CSV/JSON/notebook reading, LaTeX/figure generation, validation). **No H100/GPU
model was rerun to build this SI.**

## Publication-facing vs excluded outputs

- **Reported (publication-facing):** the Beta-Binomial layer — primary, Lexibank
  replication, positional (initial/final), within/between, matched-sample family-only
  and family+macroarea, hemisphere, Southern-macroarea, leave-family-out, leave-top-two.
- **Superseded (excluded from reporting):** the earlier Binomial-likelihood inferential
  fits. They are retained only as the Table S8 / Figure S2 sensitivity comparison and
  are labelled `superseded` in `generated/notebook_inventory.csv`.
- **Exploratory:** e.g. the prior predictive check (labelled `exploratory`).

`generated/notebook_inventory.csv` classifies every notebook cell
(`primary`/`replication`/`robustness`/`diagnostic`/`exploratory`/`superseded`/`infrastructure`).

## Rebuilding a single table or figure

```bash
cd supplement/scripts
python3 build_si_tables.py     # regenerates ALL tables/tableSN.tex
python3 build_si_figures.py    # regenerates Figures S1 (copied), S2, S3
```
Each table function in `build_si_tables.py` (`build_S1` … `build_S18`) is independent;
each figure function in `build_si_figures.py` (`figure_S1/2/3`) writes both a PDF/PNG
and, for S2/S3, a `*_source.csv`.

## Running validation

```bash
cd supplement/scripts
python3 validate_manuscript_consistency.py
```
Produces two reports in `generated/`:

- **`validation_report.md`** — repository-internal + evidence checks (counts,
  coefficients, CI endpoints, leave-family-out counts, leave-top-two contrast,
  convergence diagnostics, generated-table cell spot checks). Ends in `PASS` or an
  enumerated discrepancy list.
- **`manuscript_consistency_report.md`** — three layers reported separately:
  (a) repository-internal (manifest ↔ CSV), (b) manuscript-contract (Section-5
  declared ↔ CSV), (c) **direct manuscript-source validation = NOT PERFORMED**
  (the manuscript `.tex`/`.bib` are not in this repository).

Integer counts use exact equality; floats use the documented per-kind tolerances in
`checkpoints.py` (`coefficient` 1.5e-3, `fraction_pct` 2e-3, …). On disagreement the
validator enumerates the conflicting files/values and does **not** auto-correct.

### Known flagged item (requires human correction in the manuscript text)

The macroarea attenuation is **48.0%** (median-based, from
`asjp_beta_binomial_macroarea_matched_comparison.csv`,
`attenuation_relative_to_no_macro = 0.4797`). Any manuscript occurrence of **47.6%**
or **47.7%** is flagged for human correction; this is not rounding variation.

## Compiling the PDF

```bash
cd supplement
bash scripts/compile_supplement.sh      # latexmk -> supplement.pdf, prints diagnostics
```
The script reports LaTeX errors, undefined references/citations, missing
figures/inputs, overfull-box count, and duplicate labels. Current status: 0 errors,
0 undefined references/citations, 3 sub-4 pt overfull boxes, 20 pages.

## Where full ledgers are stored

- `generated/notebook_inventory.csv` — per-cell inventory of the executed notebook.
- `generated/source_ledger.csv` — one row per SI table/figure → Result CSV → cell.
- `generated/claim_to_evidence_map.csv` — manuscript claim → SI item → CSV → cell.
- `generated/manifest.json`, `generated/manuscript_contract.json` (+ schema).
- The complete 260-language macroarea exclusion ledger is
  `../results/asjp_macroarea_dropped_languages.csv`; the ASJP `Bookkeeping`
  pseudo-family audit is `../results/audit_asjp_bookkeeping_family.csv`.

## Directory layout

```
supplement/
  supplement.tex        supplement.pdf        supplement.bib      README.md   manifest.json
  tables/     tableS1.tex … tableS18.tex      (generated)
  figures/    figureS1..S3.{pdf,png} + *_source.csv
  generated/  notebook_inventory.csv  source_ledger.csv  claim_to_evidence_map.csv
              manifest.json  manuscript_contract.json  manuscript_contract_schema.json
              validation_report.md  manuscript_consistency_report.md
  scripts/    inventory_notebook.py  build_manifest.py  build_source_ledger.py
              build_si_tables.py  build_si_figures.py  validate_manuscript_consistency.py
              compile_supplement.sh  si_common.py  si_registry.py  checkpoints.py
```
