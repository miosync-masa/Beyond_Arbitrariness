# Beyond Arbitrariness

## Detecting Geographic Structure — and the Limits of Current Linguistic Databases

This repository contains the analysis code, executed Bayesian notebook,
result tables, publication figures, and Supplementary Information for the paper:

> **Beyond Arbitrariness: Detecting Geographic Structure — and the Limits of Current Linguistic Databases**

## Overview

This study asks whether the global distribution of word-final vowel occurrence
contains reproducible geographic structure, and whether currently available
cross-linguistic databases contain enough information to explain that structure.

Using two independently assembled lexical databases with partially overlapping
language coverage, we analyse:

- **ASJP** as the primary dataset
- **Lexibank** as a cross-database replication
- word-final versus word-initial vowel occurrence
- family-level clustering
- within-family and between-family geographic components
- broad macroarea adjustment
- Northern- and Southern-Hemisphere heterogeneity
- posterior predictive model fit
- family-level concentration at the upper boundary
- robustness to the removal of major contributing language families

The study does **not** claim that latitude, climate, migration, contact, or
biological adaptation causally determines phonological structure.

Absolute latitude is treated as a composite geographic coordinate rather than
a causal exposure. Family random intercepts model family-level clustering;
they are not a complete phylogenetic control.

The central conclusion is methodological:

> Current linguistic databases can reveal reproducible geographic structure,
> but do not yet provide the jointly integrated measurements required to
> identify its generating mechanism.

---

## Main Findings

### Cross-database geographic association

Under hierarchical Beta-Binomial models, absolute latitude was negatively
associated with word-final vowel occurrence in both databases:

| Dataset | Languages | Posterior median β | 95% credible interval |
|---|---:|---:|---:|
| ASJP | 10,886 | -0.151 | [-0.195, -0.105] |
| Lexibank | 5,501 | -0.138 | [-0.210, -0.063] |

The databases were assembled independently but contain partially overlapping
language coverage. Their agreement should therefore be interpreted as
cross-database replication, not as replication across statistically independent
language samples.

### Position-specificity

The geographic association is concentrated at the word-final edge:

| Quantity | Posterior median β | 95% credible interval |
|---|---:|---:|
| Word-initial slope | +0.028 | [-0.006, +0.063] |
| Word-final slope | -0.285 | [-0.317, -0.252] |
| Final minus initial | -0.313 | [-0.342, -0.284] |

The initial–final contrast is estimated within languages using both family-level
and language-level random intercepts.

### Family-level decomposition

| Component | Posterior median β | 95% credible interval |
|---|---:|---:|
| Between-family | -0.348 | [-0.431, -0.265] |
| Within-family | -0.033 | [-0.055, -0.011] |

Both components are negative, but the between-family component is substantially
larger. The result should not be described as genealogy-free.

### Broad macroarea adjustment

On the identical 10,626-language matched sample:

| Model | Posterior median β | 95% credible interval |
|---|---:|---:|
| Family only | -0.138 | [-0.183, -0.092] |
| Family + macroarea | -0.072 | [-0.123, -0.021] |

Broad macroarea adjustment attenuates the posterior-median coefficient by
**48.0%**.

This indicates that a substantial part of the global latitude coefficient
reflects coarse regional composition. Macroarea is a broad areal adjustment,
not a complete areal control.

### Southern-Hemisphere heterogeneity

The pooled Southern-Hemisphere coefficient is positive, but this is not a
universal reversal. Macroarea-specific estimates differ substantially:

- Papunesia: positive
- Africa: positive
- Australia: positive
- South America: uncertain and centred slightly below zero
- Southern Eurasia: uninformative (`n = 1`)

The pooled Southern estimate therefore compresses heterogeneous regional
patterns into a single coefficient.

### Posterior predictive model criticism

A Binomial likelihood recovers the overall mean but strongly underestimates
between-language dispersion.

The Beta-Binomial likelihood substantially improves dispersion, but still
underpredicts the proportion of languages whose observed word-final vowel
rate is exactly 1.0:

| Dataset | Observed exact-one share | Beta-Binomial PPC median |
|---|---:|---:|
| ASJP | 19.2% | 14.9% |
| Lexibank | 17.3% | 14.6% |

This upper-boundary excess is consistent with boundary inflation or unresolved
mixture structure. It is not, by itself, evidence for a discrete typological
class.

### Influence of major language families

Removing Atlantic-Congo and Austronesian does not eliminate the positional
contrast:

| Quantity | Posterior median β | 95% credible interval |
|---|---:|---:|
| Final minus initial, excluding both families | -0.242 | [-0.276, -0.208] |

Atlantic-Congo and Austronesian affect the global coefficient in opposite
directions. The global latitude estimate should therefore be interpreted as a
composite summary of multiple genealogical and geographic configurations.

---

## Repository Structure

```text
Beyond_Arbitrariness/
├── Beyond_Arbitrariness_BetaBinomial_Diagnostics_final.ipynb
├── results/
│   ├── publication-facing coefficient tables
│   ├── MCMC diagnostics
│   ├── posterior predictive summaries
│   ├── family-level audits
│   ├── macroarea sample-flow audits
│   ├── robustness analyses
│   └── figure-source CSV files
├── figures/
│   ├── main-text figures
│   └── supplementary figures
├── supplement/
│   ├── supplement.tex
│   ├── supplement.pdf
│   ├── tables/
│   ├── figures/
│   ├── generated/
│   └── scripts/
├── scripts/
│   └── earlier exploratory and frequentist analyses
└── README.md
```

### Publication-facing analysis

The primary analysis source is:

```text
Beyond_Arbitrariness_BetaBinomial_Diagnostics_final.ipynb
```

The executed notebook contains:

- data acquisition and commit hashes
- phoneme and vowel classification
- language-level aggregation
- isolate/unknown family recoding
- hierarchical model definitions
- prior specifications
- NUTS configuration
- posterior summaries
- posterior predictive checks
- robustness analyses
- output generation
- reproducibility metadata

The numerical source of truth for publication-facing results is the collection
of generated CSV files in:

```text
results/
```

The notebook provides analysis context and execution history; the Result CSVs
provide authoritative numerical values.

### Earlier scripts

The scripts in the repository root `scripts/` directory belong to an earlier
exploratory and frequentist analysis pipeline.

They document the development history of the project but are not the primary
source for the hierarchical Beta-Binomial results reported in the current paper.

---

## Data Sources

| Database | Role | Primary analysis sample | Transcription | Source |
|---|---|---:|---|---|
| ASJP | Primary analysis | 10,886 languages | ASJPcode | https://github.com/lexibank/asjp |
| Lexibank | Cross-database replication | 5,501 languages | IPA / CLTS segments | https://github.com/lexibank/lexibank-analysed |
| Glottolog | Genealogical metadata | — | Classification metadata | https://glottolog.org |
| WALS | Typological precedent | 486 languages in the syllable-structure chapter | Categorical typology | https://wals.info/chapter/12 |

The ASJP pre-filter language summary contained 11,393 languages.
After requiring at least 20 attested concepts, the primary inferential sample
contained 10,886 languages and 646 represented family identifiers.

Raw source databases are not redistributed in this repository.
Exact source commit hashes are recorded in:

```text
results/reproducibility_metadata.json
```

---

## Data Setup

The executed notebook clones the required lexical databases automatically.

For a manual setup:

```bash
git clone https://github.com/miosync-masa/Beyond_Arbitrariness.git
git clone https://github.com/lexibank/asjp.git
git clone https://github.com/lexibank/lexibank-analysed.git
```

A typical sibling-directory layout is:

```text
working-directory/
├── Beyond_Arbitrariness/
├── asjp/
└── lexibank-analysed/
```

Paths used in the notebook may need to be adjusted outside Google Colab.

---

## Running the Analysis

### Open the final notebook in Google Colab

[Open the final Beta-Binomial notebook in Colab](https://colab.research.google.com/github/miosync-masa/Beyond_Arbitrariness/blob/main/Beyond_Arbitrariness_BetaBinomial_Diagnostics_final.ipynb)

### Important computational note

The complete Bayesian pipeline uses large hierarchical NumPyro models and was
originally executed on a GPU runtime.

The committed notebook outputs and Result CSVs are sufficient to inspect,
audit, compile the Supplementary Information, and regenerate publication tables
without rerunning the expensive posterior models.

A full refit is computationally expensive and is not required for ordinary
repository inspection.

---

## Software

The recorded analysis environment includes:

- Python
- JAX 0.7.2
- NumPyro 0.19.0
- ArviZ
- pandas
- NumPy
- SciPy
- matplotlib
- pyarrow
- netCDF4

The notebook installation cell uses:

```bash
pip install \
  "numpyro>=0.16,<0.20" \
  "arviz>=0.20,<0.23" \
  "pandas>=2.2,<3" \
  "pyarrow>=17,<22" \
  "matplotlib>=3.9,<4" \
  "scipy>=1.13,<2" \
  "netCDF4>=1.7"
```

Only JAX and NumPyro exact runtime versions were printed during execution.
The remaining entries above are installation constraints, not necessarily exact
runtime versions.

The master random seed was:

```text
20260711
```

Repository, ASJP, and Lexibank commit hashes used for the reported analysis are
stored in:

```text
results/reproducibility_metadata.json
```

---

## Supplementary Information

The `supplement/` directory contains an independently compilable
Elsevier-compatible Supplementary Information package for submission to
*Language Sciences*.

It includes:

- data provenance and preprocessing
- sample-flow and identifier audits
- complete prior and model specifications
- MCMC convergence diagnostics
- Binomial versus Beta-Binomial sensitivity
- wordlist-size robustness
- leave-family-out analyses
- family-level posterior predictive residuals
- macroarea and hemispheric heterogeneity
- computational provenance
- manuscript-contract validation
- claim-to-evidence mapping

The SI is designed as an audit layer:

```text
manuscript claim
    ↓
SI table or figure
    ↓
Result CSV
    ↓
notebook cell / Python definition
```

The H100/GPU posterior models were not rerun when producing the SI.

### Rebuild the SI

```bash
python supplement/scripts/inventory_notebook.py
python supplement/scripts/build_manifest.py
python supplement/scripts/build_source_ledger.py
python supplement/scripts/build_si_tables.py
python supplement/scripts/build_si_figures.py
python supplement/scripts/validate_manuscript_consistency.py
bash supplement/scripts/compile_supplement.sh
```

See:

```text
supplement/README.md
```

for the full build and validation procedure.

---

## Interpretation Boundaries

The analyses support the following claims:

- word-final vowel occurrence contains reproducible geographic structure
- the pattern appears in both ASJP and Lexibank
- the association is concentrated at the word-final edge
- family-level and broad areal composition explain substantial parts of the pattern
- a residual conditional association remains
- a one-component Beta-Binomial does not capture the entire observed distribution
- current databases do not uniquely identify the generating mechanism

The analyses do **not** establish:

- a causal effect of latitude
- climatic adaptation
- migration history
- contact corridors
- biological causation
- a universal Northern/Southern reversal
- a discrete open-syllable class
- full phylogenetic independence
- statistical independence of the two databases

Word-final vowel occurrence is measured directly.
Open-syllable structure is a plausible typological correlate, but the two are
not treated as equivalent.

---

## Reproducibility and Audit Files

Important audit outputs include:

```text
results/reproducibility_metadata.json
results/asjp_macroarea_sample_flow.csv
results/asjp_macroarea_sample_difference.csv
results/asjp_unknown_family_split_audit.csv
results/asjp_beta_binomial_macroarea_matched_comparison.csv
results/paper1_beta_binomial_final_effects.csv
results/asjp_beta_binomial_ppc_summary.csv
results/lexibank_beta_binomial_ppc_summary.csv
results/asjp_boundary_family_exclusion_sensitivity.csv
results/asjp_position_excluding_top_two_boundary_families.csv
```

The SI additionally generates:

```text
supplement/manifest.json
supplement/generated/notebook_inventory.csv
supplement/generated/source_ledger.csv
supplement/generated/claim_to_evidence_map.csv
supplement/generated/manuscript_contract.json
supplement/generated/validation_report.md
supplement/generated/manuscript_consistency_report.md
```

---

## License

The analysis code and generated outputs are released under the license specified
in this repository.

ASJP, Lexibank, Glottolog, WALS, and other source datasets remain subject to
their respective licenses and citation requirements.

---

## Citation

Citation information will be added upon publication.

Until then, please cite the repository and the relevant source databases when
reusing the analysis pipeline or generated results.
