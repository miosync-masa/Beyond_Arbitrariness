# Manuscript consistency report (three layers)

The manuscript `.tex`/`.bib` are **not** present in this repository. The three consistency layers below are therefore reported **separately** and must not be conflated.

## Layer (a) - Repository-internal consistency (manifest <-> Result CSV)

**PASS** - 28 manifest values, 3 CI pairs, 3 leave-family-out counts, 8 diagnostics, 6 table spot checks; 0 failure(s). Every publication-facing number in the SI resolves to a Result CSV.

## Layer (b) - Manuscript-contract consistency (Section-5 declared <-> Result CSV)

**1 item(s) FLAGGED for human correction**

| checkpoint | manuscript declared | authoritative CSV | tol | status |
|---|---|---|---|---|
| asjp_prefilter_languages | 11393 | 11393.0 | 0.0 | PASS |
| asjp_prefilter_family_ids | 667 | 667.0 | 0.0 | PASS |
| asjp_primary_languages | 10886 | 10886.0 | 0.0 | PASS |
| asjp_primary_family_ids | 646 | 646.0 | 0.0 | PASS |
| lexibank_languages | 5501 | 5501.0 | 0.0 | PASS |
| macroarea_matched_languages | 10626 | 10626.0 | 0.0 | PASS |
| macroarea_matched_family_ids | 386 | 386.0 | 0.0 | PASS |
| macroarea_unlinked_languages | 260 | 260.0 | 0.0 | PASS |
| primary_beta_lat | -0.151 | -0.150881 | 0.0015 | PASS |
| lexibank_beta_lat | -0.138 | -0.138695 | 0.0015 | PASS |
| beta_initial | 0.028 | 0.028522 | 0.0015 | PASS |
| beta_final | -0.285 | -0.284989 | 0.0015 | PASS |
| beta_final_minus_initial | -0.313 | -0.313191 | 0.0015 | PASS |
| beta_between | -0.348 | -0.348012 | 0.0015 | PASS |
| beta_within | -0.033 | -0.032783 | 0.0015 | PASS |
| matched_no_macroarea_beta | -0.138 | -0.137723 | 0.0015 | PASS |
| matched_macroarea_beta | -0.072 | -0.071663 | 0.0015 | PASS |
| macroarea_attenuation | 0.476 | 0.479662 | 0.002 | FLAG-FOR-HUMAN-CORRECTION |
| north_beta | -0.23 | -0.230113 | 0.0015 | PASS |
| south_beta | 0.527 | 0.525923 | 0.0015 | PASS |
| leave_both_final_minus_initial | -0.242 | -0.242169 | 0.0015 | PASS |

### Flagged items (require human correction in the manuscript text)

- **macroarea_attenuation**: manuscript declares `0.476`, authoritative CSV value is `0.479662` (tolerance 0.002). Manuscript text states 47.6%; authoritative median-based CSV value is 0.4797 (48.0%). Publication-facing value per approved correction = 48.0%. FLAG 47.6/47.7% for human correction.
  Conflicting sources: manuscript text vs `asjp_beta_binomial_macroarea_matched_comparison.csv` (`attenuation_relative_to_no_macro`). Recommended inspection: manuscript occurrences of 47.6% / 47.7%; replace with 48.0% (median-based). Not auto-corrected.

## Layer (c) - Direct manuscript-source validation

**NOT PERFORMED.** The external manuscript `.tex` and `.bib` are not in this repository, so publication-facing values cannot be parsed directly from the manuscript source. The Section-5 checkpoint values (Layer b) are used as the manuscript contract in their place. If the manuscript source is later added, re-point the validator at parsed `.tex` values to perform this layer.
