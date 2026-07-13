#!/usr/bin/env python3
"""
build_si_figures.py
===================
Produce the three supplementary figures:

* Figure S1 -- copied verbatim from the existing final figure
  (figures/figureS1_family_boundary_ppc.{pdf,png}); NOT redrawn.
* Figure S2 -- Binomial vs Beta-Binomial posterior-predictive comparison, built
  from the four *_ppc_summary.csv files.  Per approved correction #4, this is a
  comparison of summary statistics (overall mean, between-language SD, exact-one
  share); NO per-bin distribution is reconstructed from summaries, because no
  per-bin Binomial predictive output exists in the committed artifacts.
* Figure S3 -- boundary mass across coverage thresholds, from
  boundary_mass_by_coverage.csv.

Each generated figure also writes a tidy figure-source CSV.  Uses only existing
result / predictive-summary files; no model is rerun.
"""
from __future__ import annotations

import csv
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import si_common as C

# restrained, colour-blind-safe palette (no invented semantic colours)
COL_OBS = "#222222"      # observed = near-black
COL_BINOM = "#B0B0B0"    # Binomial = grey (the superseded layer)
COL_BB = "#4C72B0"       # Beta-Binomial = muted blue (the reported layer)
PLT_RC = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
          "axes.titlesize": 10, "figure.dpi": 150}


def save(fig, stem):
    pdf = os.path.join(C.SUPFIG, stem + ".pdf")
    png = os.path.join(C.SUPFIG, stem + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
def figure_S1():
    outs = []
    for ext in ("pdf", "png"):
        src = os.path.join(C.FIGURES, f"figureS1_family_boundary_ppc.{ext}")
        if os.path.exists(src):
            dst = os.path.join(C.SUPFIG, f"figureS1_family_boundary_ppc.{ext}")
            shutil.copyfile(src, dst)
            outs.append(dst)
    # figure-source CSV already exists as results/figureS1_family_boundary_ppc.csv; copy for locality
    src_csv = C.csv_path("figureS1_family_boundary_ppc.csv")
    shutil.copyfile(src_csv, os.path.join(C.SUPFIG, "figureS1_source.csv"))
    return outs


# ---------------------------------------------------------------------------
STAT_LABELS = [("mean_language_rate", "Overall mean\n(mean language rate)"),
               ("sd_language_rate", "Between-language\ndispersion (SD)"),
               ("share_rate_one", "Exact-one\nboundary share")]


def _load_ppc(binom_csv, bb_csv):
    binom = {r["statistic"]: r for r in C.load_csv(binom_csv)}
    bb = {r["statistic"]: r for r in C.load_csv(bb_csv)}
    return binom, bb


def figure_S2():
    panels = [
        ("ASJP", "asjp_primary_final_ppc_summary.csv", "asjp_beta_binomial_ppc_summary.csv"),
        ("Lexibank", "lexibank_final_ppc_summary.csv", "lexibank_beta_binomial_ppc_summary.csv"),
    ]
    # figure-source CSV
    rows_src = []
    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=False)
        fig.subplots_adjust(wspace=0.28)
        for ax, (db, binom_csv, bb_csv) in zip(axes, panels):
            binom, bb = _load_ppc(binom_csv, bb_csv)
            x = range(len(STAT_LABELS))
            w = 0.27
            obs = [float(bb[s]["observed"]) for s, _ in STAT_LABELS]
            bm = [float(binom[s]["ppc_median"]) for s, _ in STAT_LABELS]
            bm_lo = [float(binom[s]["ppc_median"]) - float(binom[s]["ppc_2.5"]) for s, _ in STAT_LABELS]
            bm_hi = [float(binom[s]["ppc_97.5"]) - float(binom[s]["ppc_median"]) for s, _ in STAT_LABELS]
            bbm = [float(bb[s]["ppc_median"]) for s, _ in STAT_LABELS]
            bb_lo = [float(bb[s]["ppc_median"]) - float(bb[s]["ppc_2.5"]) for s, _ in STAT_LABELS]
            bb_hi = [float(bb[s]["ppc_97.5"]) - float(bb[s]["ppc_median"]) for s, _ in STAT_LABELS]
            ax.bar([i - w for i in x], obs, w, color=COL_OBS, label="Observed")
            ax.bar(list(x), bm, w, color=COL_BINOM, yerr=[bm_lo, bm_hi], capsize=2,
                   ecolor="#555555", label="Binomial PPC")
            ax.bar([i + w for i in x], bbm, w, color=COL_BB, yerr=[bb_lo, bb_hi], capsize=2,
                   ecolor="#33456b", label="Beta-Binomial PPC")
            ax.set_xticks(list(x))
            ax.set_xticklabels([lbl for _, lbl in STAT_LABELS], fontsize=7.5)
            ax.set_title(db)
            ax.set_ylim(0, max(max(obs), max(bm), max(bbm)) * 1.18)
            for s, _ in STAT_LABELS:
                rows_src.append({
                    "database": db, "statistic": s,
                    "observed": bb[s]["observed"],
                    "binomial_ppc_median": binom[s]["ppc_median"],
                    "binomial_ppc_2.5": binom[s]["ppc_2.5"], "binomial_ppc_97.5": binom[s]["ppc_97.5"],
                    "beta_binomial_ppc_median": bb[s]["ppc_median"],
                    "beta_binomial_ppc_2.5": bb[s]["ppc_2.5"], "beta_binomial_ppc_97.5": bb[s]["ppc_97.5"],
                })
        axes[0].set_ylabel("Proportion")
        axes[1].legend(frameon=False, fontsize=8, loc="upper right")
        fig.suptitle("Binomial vs Beta-Binomial posterior predictive checks", fontsize=11)
        outs = save(fig, "figureS2_binomial_vs_betabinomial_ppc")

    src_csv = os.path.join(C.SUPFIG, "figureS2_source.csv")
    with open(src_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_src[0].keys()))
        w.writeheader()
        w.writerows(rows_src)
    return list(outs) + [src_csv]


# ---------------------------------------------------------------------------
def figure_S3():
    rows = C.load_csv("boundary_mass_by_coverage.csv")
    dbs = ["ASJP", "Lexibank"]
    series = [("share_exact_one", "Exact $=1.00$", "o", "-", "#4C72B0"),
              ("share_ge_0_99", r"$\geq 0.99$", "s", "--", "#DD8452"),
              ("share_ge_0_95", r"$\geq 0.95$", "^", ":", "#55A868")]
    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
        for ax, db in zip(axes, dbs):
            dbrows = sorted([r for r in rows if r["database"] == db],
                            key=lambda r: int(r["min_n_total"]))
            xs = [int(r["min_n_total"]) for r in dbrows]
            for col, lbl, mk, ls, colr in series:
                ys = [float(r[col]) for r in dbrows]
                ax.plot(xs, ys, marker=mk, linestyle=ls, color=colr, label=lbl)
            ax.set_title(db)
            ax.set_xlabel("Minimum total word count")
            ax.set_xticks(xs)
            ax.grid(True, axis="y", alpha=0.25)
        axes[0].set_ylabel("Share of languages")
        axes[1].legend(frameon=False, fontsize=8, title="Boundary")
        fig.suptitle("Boundary mass across coverage thresholds", fontsize=11)
        outs = save(fig, "figureS3_boundary_mass_coverage")

    # tidy figure-source CSV (copy of the authoritative sweep)
    src_csv = os.path.join(C.SUPFIG, "figureS3_source.csv")
    shutil.copyfile(C.csv_path("boundary_mass_by_coverage.csv"), src_csv)
    return list(outs) + [src_csv]


def main() -> int:
    for name, fn in [("Figure S1 (copied)", figure_S1),
                     ("Figure S2", figure_S2),
                     ("Figure S3", figure_S3)]:
        outs = fn()
        print(f"{name}:")
        for o in outs:
            print("   ", os.path.relpath(o, C.REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
