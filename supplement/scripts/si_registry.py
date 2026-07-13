#!/usr/bin/env python3
"""
si_registry.py
==============
Single registry describing every SI table and figure: its section, title,
authoritative Result CSV input(s), the notebook cell(s) / Python source that
produced those inputs, the model name, the sample definition, the publication
status, and a short consistency note.

This registry drives:
  * build_source_ledger.py  (source_ledger.csv + claim_to_evidence_map.csv)
  * build_si_tables.py      (metadata for captions / labels)

Notebook cell indices refer to the executed master notebook
``Beyond_Arbitrariness_BetaBinomial_Diagnostics_final.ipynb`` (see
notebook_inventory.csv).
"""

NOTEBOOK = "Beyond_Arbitrariness_BetaBinomial_Diagnostics_final.ipynb"

# publication_status vocabulary: primary | replication | robustness | diagnostic | audit | provenance
SI_ITEMS = [
    dict(id="S1", kind="table", section="S1", table_or_figure="Table S1",
         title="Database provenance and preprocessing summary",
         csvs=["asjp_family_audit.csv", "lexibank_family_audit.csv",
               "asjp_macroarea_sample_flow.csv", "asjp_family_counts.csv",
               "lexibank_family_counts.csv", "reproducibility_metadata.json"],
         cells="4,6,7,53", model="-", sample="ASJP & Lexibank language summaries",
         status="provenance",
         consistency="counts cross-checked against manifest & sample-flow CSV"),
    dict(id="S2", kind="table", section="S1", table_or_figure="Table S2",
         title="Segment and vowel classification rules",
         csvs=[], cells="6,7", model="-",
         sample="rule definitions (code, not data)", status="provenance",
         consistency="verbatim from notebook cell 6/7 classification code"),
    dict(id="S3", kind="table", section="S2", table_or_figure="Table S3",
         title="ASJP sample flow",
         csvs=["asjp_macroarea_sample_flow.csv", "asjp_macroarea_sample_difference.csv",
               "asjp_unknown_family_split_audit.csv"],
         cells="67,99", model="-", sample="ASJP prefilter -> primary -> macroarea-matched",
         status="audit", consistency="exact counts vs manifest"),
    dict(id="S4", kind="table", section="S2", table_or_figure="Table S4",
         title="Macroarea-linked sample composition",
         csvs=["asjp_macroarea_distribution.csv"], cells="99", model="-",
         sample="macroarea complete-case (10,626 languages)", status="audit",
         consistency="family/language counts vs matched sample (386 families)"),
    dict(id="S5", kind="table", section="S2", table_or_figure="Table S5",
         title="Macroarea linkage audit summary",
         csvs=["asjp_macroarea_exclusion_counts.csv", "asjp_macroarea_source_metadata.csv"],
         cells="99", model="-", sample="linkage audit (260 unlinked)", status="audit",
         consistency="ledger of 260 unlinked retained as asjp_macroarea_dropped_languages.csv"),
    dict(id="S6", kind="table", section="S3", table_or_figure="Table S6",
         title="Prior specification and parameterisation",
         csvs=[], cells="11,25,70", model="all", sample="model specifications (code)",
         status="provenance", consistency="priors read from notebook code, not posteriors"),
    dict(id="S7", kind="table", section="S4", table_or_figure="Table S7",
         title="Complete MCMC diagnostics",
         csvs=["asjp_beta_binomial_primary_diagnostics.csv",
               "lexibank_beta_binomial_replication_diagnostics.csv",
               "asjp_beta_binomial_within_between_diagnostics.csv",
               "asjp_beta_binomial_position_interaction_diagnostics.csv",
               "asjp_beta_binomial_hemisphere_diagnostics.csv",
               "asjp_beta_binomial_macroarea_control_diagnostics.csv",
               "asjp_beta_binomial_macroarea_matched_no_macroarea_diagnostics.csv",
               "asjp_beta_binomial_south_macroarea_slopes_diagnostics.csv"],
         cells="72,74,76,78,80,85,86,95,97,100", model="all Beta-Binomial reported models",
         sample="per-model", status="diagnostic",
         consistency="R-hat/ESS/divergences straight from *_diagnostics.csv"),
    dict(id="S8", kind="table", section="S5", table_or_figure="Table S8",
         title="Binomial versus Beta-Binomial coefficient comparison",
         csvs=["asjp_primary_effect.csv", "asjp_beta_binomial_primary_effect.csv",
               "asjp_initial_final_effects.csv", "asjp_beta_binomial_position_effects.csv",
               "asjp_within_between_effects.csv", "asjp_beta_binomial_within_between_effects.csv"],
         cells="13,72,36,76,30,74", model="Binomial (superseded) vs Beta-Binomial (reported)",
         sample="ASJP primary", status="robustness",
         consistency="paired coefficients from both likelihood layers"),
    dict(id="S9", kind="table", section="S6", table_or_figure="Table S9",
         title="Boundary mass by minimum total word count",
         csvs=["boundary_mass_by_coverage.csv"], cells="89", model="-",
         sample="coverage sweep (min total word count)", status="robustness",
         consistency="axis is min_n_total (word count), NOT concept count; inclusion criterion is >=20 concepts"),
    dict(id="S10", kind="table", section="S7", table_or_figure="Table S10",
         title="Leave-family-out primary estimates",
         csvs=["asjp_beta_binomial_primary_effect.csv",
               "asjp_boundary_family_exclusion_sensitivity.csv"],
         cells="72,95", model="primary_beta_binomial_model", sample="ASJP primary & leave-family-out",
         status="robustness", consistency="beta_lat medians vs manifest"),
    dict(id="S11", kind="table", section="S7", table_or_figure="Table S11",
         title="Positional estimates after excluding Atlantic-Congo and Austronesian",
         csvs=["asjp_beta_binomial_position_effects.csv",
               "asjp_position_excluding_top_two_boundary_families.csv"],
         cells="76,97", model="position_beta_binomial_model",
         sample="ASJP primary & leave-top-two", status="robustness",
         consistency="final-minus-initial -0.242 vs manifest"),
    dict(id="S12", kind="table", section="S8", table_or_figure="Table S12",
         title="Largest contributors to exact-one boundary mass",
         csvs=["asjp_family_boundary_concentration.csv", "lexibank_family_boundary_concentration.csv"],
         cells="91", model="-", sample="ASJP & Lexibank family boundary concentration",
         status="diagnostic", consistency="Bookkeeping pseudo-family excluded from interpretive rows, retained in ledger"),
    dict(id="S13", kind="table", section="S8", table_or_figure="Table S13",
         title="Family-level posterior predictive residuals",
         csvs=["asjp_beta_binomial_family_boundary_ppc.csv",
               "lexibank_beta_binomial_family_boundary_ppc.csv"],
         cells="93", model="primary_beta_binomial_model (PPC)",
         sample="major families in both databases", status="diagnostic",
         consistency="observed-minus-PPC residuals from *_family_boundary_ppc.csv"),
    dict(id="S14", kind="table", section="S9", table_or_figure="Table S14",
         title="Matched-sample macroarea adjustment",
         csvs=["asjp_beta_binomial_primary_effect.csv",
               "asjp_beta_binomial_macroarea_matched_comparison.csv"],
         cells="72,85,100", model="matched_sample / macroarea_beta_binomial_model",
         sample="full primary & matched (10,626)", status="robustness",
         consistency="attenuation 48.0% (median-based) vs manifest; 47.6/47.7% flagged"),
    dict(id="S15", kind="table", section="S9", table_or_figure="Table S15",
         title="Southern-Hemisphere macroarea-specific slopes",
         csvs=["asjp_beta_binomial_south_macroarea_effects.csv"], cells="86", model="south_macroarea_beta_binomial_model",
         sample="Southern-Hemisphere macroareas", status="robustness",
         consistency="Southern Eurasia n=1 marked uninformative"),
    dict(id="S16", kind="table", section="S10", table_or_figure="Table S16",
         title="Cross-database language overlap",
         csvs=["asjp_family_audit.csv", "lexibank_family_audit.csv"], cells="53",
         model="-", sample="ASJP vs Lexibank language sets", status="audit",
         consistency="NOT COMPUTABLE from committed artifacts: no shared identifier / Glottocode / per-language Lexibank file in repo"),
    dict(id="S17", kind="table", section="S11", table_or_figure="Table S17",
         title="Computational environment",
         csvs=["reproducibility_metadata.json"], cells="2,3", model="-",
         sample="environment", status="provenance",
         consistency="recorded runtime versions vs pip constraints kept distinct"),
    dict(id="S18", kind="table", section="S11", table_or_figure="Table S18",
         title="Analysis provenance and output-file map",
         csvs=["reproducibility_metadata.json"], cells="4,72,80,85,86,95,97,100",
         model="all", sample="per-analysis", status="provenance",
         consistency="derived from source_ledger.csv + manifest"),

    # ---- figures ----
    dict(id="FS1", kind="figure", section="S8", table_or_figure="Figure S1",
         title="Family-level boundary concentration and model misfit are not equivalent",
         csvs=["figureS1_family_boundary_ppc.csv"], cells="106",
         model="primary_beta_binomial_model (PPC)", sample="major families, both databases",
         status="diagnostic", consistency="existing final figure reused verbatim (copied, not redrawn)"),
    dict(id="FS2", kind="figure", section="S5", table_or_figure="Figure S2",
         title="Binomial versus Beta-Binomial posterior predictive comparison",
         csvs=["asjp_primary_final_ppc_summary.csv", "asjp_beta_binomial_ppc_summary.csv",
               "lexibank_final_ppc_summary.csv", "lexibank_beta_binomial_ppc_summary.csv"],
         cells="59,82", model="Binomial vs Beta-Binomial PPC",
         sample="ASJP & Lexibank", status="diagnostic",
         consistency="summary-statistic comparison (mean, between-language SD, exact-one share); "
                     "NO per-bin distribution reconstructed from summaries"),
    dict(id="FS3", kind="figure", section="S6", table_or_figure="Figure S3",
         title="Boundary mass across coverage thresholds",
         csvs=["boundary_mass_by_coverage.csv"], cells="89", model="-",
         sample="coverage sweep (min total word count)", status="robustness",
         consistency="min total word count axis; inclusion criterion >=20 concepts stated in caption"),
]


def by_id(item_id):
    for it in SI_ITEMS:
        if it["id"] == item_id:
            return it
    raise KeyError(item_id)
