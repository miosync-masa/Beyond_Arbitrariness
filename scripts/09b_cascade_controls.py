"""
09b_cascade_controls.py
========================
Genealogical and areal controls for the phonotactic-morphological cascade.

B1) Mixed-effects logistic regression: feature ~ Vfinal + abs_lat + (1 | family)
B2) Macroarea fixed effect:             feature ~ Vfinal + abs_lat + macroarea + (1 | family)
B3) Family-mean aggregation:            Spearman correlation on family-level means
B4) Permutation by family:              Shuffle Vfinal within families, 5000 iterations

Requirements: statsmodels, numpy, scipy
Data: ASJP, Grambank, (optionally WALS) — cloned as siblings or adjust paths below.
"""

import csv
import json
import random
from collections import defaultdict
from math import sqrt, log, exp

# --- Attempt imports; fall back gracefully ---
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not found. B1/B2 will be skipped.")

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.families.links import Logit
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not found. B1/B2 will be skipped.")

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Using manual Spearman for B3.")


# ================================================================
# CONFIGURATION — adjust paths as needed
# ================================================================
ASJP_LANG = 'asjp/cldf/languages.csv'
ASJP_FORMS = 'asjp/cldf/forms.csv'
GRAMBANK_LANG = 'grambank/cldf/languages.csv'
GRAMBANK_VALUES = 'grambank/cldf/values.csv'

# Minimum thresholds
MIN_WORDS_PER_LANG = 20
MIN_LANGS_PER_FAMILY = 3      # for family-mean analysis
MIN_FAMILIES_FOR_CORR = 20    # minimum families for B3

# Permutation iterations
N_PERM = 5000

# Grambank features of interest (prefix–suffix mirror + morphological features)
GB_FEATURES = {
    # Suffix features (expected: positive with Vfinal)
    'GB080': 'Verb suffixes/enclitics (non-person)',
    'GB089': 'S-argument suffix on verb',
    'GB091': 'A-argument suffix on verb',
    'GB093': 'P-argument suffix on verb',
    # Prefix features (expected: negative with Vfinal)
    'GB079': 'Verb prefixes/proclitics (non-person)',
    'GB090': 'S-argument prefix on verb',
    'GB092': 'A-argument prefix on verb',
    'GB094': 'P-argument prefix on verb',
    # Morphological features
    'GB070': 'Morphological case (non-pronominal)',
    'GB082': 'Present tense marking on verbs',
    'GB083': 'Past tense marking on verbs',
    'GB084': 'Future tense marking on verbs',
    'GB086': 'Perfective/imperfective distinction',
    'GB133': 'Verb-final order',
    'GB136': 'Fixed constituent order',
    'GB044': 'Productive plural marking on nouns',
}

VOWELS = set('ieE3auo')
CONSONANTS = set('pbfvmw8tdszclnrSZCjT5ykgxNqXh7L4G!')


# ================================================================
# STEP 1: Load ASJP → compute per-language word-final V%
# ================================================================
def load_asjp_wordfinal():
    """Compute word-final vowel % per ASJP language, indexed by Glottocode."""
    languages = {}
    with open(ASJP_LANG, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            lat = float(row['Latitude']) if row.get('Latitude') else None
            languages[row['ID']] = {
                'glottocode': row.get('Glottocode', ''),
                'name': row.get('Name', ''),
                'family': row.get('Family', ''),
                'lat': lat,
                'abs_lat': abs(lat) if lat is not None else None,
            }

    lang_finals = defaultdict(lambda: {'v': 0, 'c': 0})
    with open(ASJP_FORMS, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('Loan') == 'true':
                continue
            form = row['Form']
            phonemes = [ch for ch in form if ch in VOWELS or ch in CONSONANTS]
            if not phonemes:
                continue
            final = phonemes[-1]
            lang_id = row['Language_ID']
            if final in VOWELS:
                lang_finals[lang_id]['v'] += 1
            else:
                lang_finals[lang_id]['c'] += 1

    lang_vfinal = {}
    for lang_id, counts in lang_finals.items():
        total = counts['v'] + counts['c']
        if total >= MIN_WORDS_PER_LANG:
            info = languages.get(lang_id, {})
            gc = info.get('glottocode', '')
            if gc and info.get('abs_lat') is not None:
                lang_vfinal[gc] = {
                    'vfinal_pct': counts['v'] / total * 100,
                    'vfinal_prop': counts['v'] / total,  # 0–1 for models
                    'family': info.get('family', ''),
                    'abs_lat': info['abs_lat'],
                    'name': info.get('name', ''),
                    'n_words': total,
                }
    return lang_vfinal


# ================================================================
# STEP 2: Load Grambank features + macroarea
# ================================================================
def load_grambank():
    """Load Grambank binary features and macroarea, indexed by Glottocode."""
    # Load languages for macroarea
    gb_langs = {}
    with open(GRAMBANK_LANG, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            gc = row.get('Glottocode', row.get('ID', ''))
            macroarea = row.get('Macroarea', row.get('macroarea', ''))
            gb_langs[gc] = macroarea if macroarea else 'Unknown'

    # Load values
    gb_data = defaultdict(dict)
    with open(GRAMBANK_VALUES, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            gc = row['Language_ID']
            feat = row['Parameter_ID']
            val = row['Value']
            if val in ('0', '1'):
                gb_data[gc][feat] = int(val)

    return gb_data, gb_langs


# ================================================================
# HELPER: Manual Spearman (fallback if no scipy)
# ================================================================
def rank_data(x):
    """Simple ranking."""
    indexed = sorted(enumerate(x), key=lambda t: t[1])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j+1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson_r(x, y):
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    sx = sqrt(sum((xi - mx)**2 for xi in x) / (n - 1))
    sy = sqrt(sum((yi - my)**2 for yi in y) / (n - 1))
    if sx == 0 or sy == 0:
        return 0.0, 1.0
    r = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / ((n - 1) * sx * sy)
    # t-test for significance
    if abs(r) >= 1.0:
        return r, 0.0
    t_stat = r * sqrt(n - 2) / sqrt(1 - r**2)
    # Two-tailed p via t-distribution approximation
    df = n - 2
    p = 2 * t_survival(abs(t_stat), df)
    return r, p


def t_survival(t, df):
    """Rough approximation of P(T > t) for t-distribution."""
    # Use normal approximation for df > 30
    if df > 30:
        from math import erfc
        return 0.5 * erfc(t / sqrt(2))
    # For smaller df, use a crude approximation
    x = df / (df + t * t)
    return 0.5 * x ** (df / 2)


def spearman_manual(x, y):
    rx = rank_data(x)
    ry = rank_data(y)
    return pearson_r(rx, ry)


# ================================================================
# B1: Mixed-effects logistic regression (LPM approximation)
# ================================================================
def run_b1(data, gb_data, feat_id):
    """
    feature ~ vfinal_prop + abs_lat + (1 | family)
    Uses linear mixed model as approximation for binary outcome.
    Returns: (coef_vfinal, se, z, p, n, n_families)
    """
    if not HAS_NUMPY or not HAS_STATSMODELS:
        return None

    rows = []
    for gc in data:
        if gc in gb_data and feat_id in gb_data[gc]:
            rows.append({
                'feature': gb_data[gc][feat_id],
                'vfinal': data[gc]['vfinal_prop'],
                'abs_lat': data[gc]['abs_lat'],
                'family': data[gc]['family'],
            })

    if len(rows) < 50:
        return None

    import pandas as pd
    df = pd.DataFrame(rows)

    # Drop families with < 2 languages (can't estimate random effect)
    fam_counts = df['family'].value_counts()
    valid_fams = fam_counts[fam_counts >= 2].index
    df = df[df['family'].isin(valid_fams)]

    if len(df) < 50 or df['family'].nunique() < 10:
        return None

    try:
        model = smf.mixedlm(
            "feature ~ vfinal + abs_lat",
            data=df,
            groups=df["family"],
        )
        result = model.fit(reml=True, method='lbfgs')

        coef = result.params.get('vfinal', None)
        se = result.bse.get('vfinal', None)
        z = result.tvalues.get('vfinal', None)
        p = result.pvalues.get('vfinal', None)

        if coef is None:
            return None

        return {
            'coef': coef,
            'se': se,
            'z': z,
            'p': p,
            'n': len(df),
            'n_families': df['family'].nunique(),
            'direction': '+' if coef > 0 else '-',
        }
    except Exception as e:
        print(f"    B1 model failed for {feat_id}: {e}")
        return None


# ================================================================
# B2: Macroarea fixed effect
# ================================================================
def run_b2(data, gb_data, gb_langs, feat_id):
    """
    feature ~ vfinal_prop + abs_lat + macroarea + (1 | family)
    """
    if not HAS_NUMPY or not HAS_STATSMODELS:
        return None

    rows = []
    for gc in data:
        if gc in gb_data and feat_id in gb_data[gc]:
            macroarea = gb_langs.get(gc, 'Unknown')
            rows.append({
                'feature': gb_data[gc][feat_id],
                'vfinal': data[gc]['vfinal_prop'],
                'abs_lat': data[gc]['abs_lat'],
                'family': data[gc]['family'],
                'macroarea': macroarea,
            })

    if len(rows) < 50:
        return None

    import pandas as pd
    df = pd.DataFrame(rows)

    # Drop tiny families
    fam_counts = df['family'].value_counts()
    valid_fams = fam_counts[fam_counts >= 2].index
    df = df[df['family'].isin(valid_fams)]

    # Drop macroarea with < 5 languages
    area_counts = df['macroarea'].value_counts()
    valid_areas = area_counts[area_counts >= 5].index
    df = df[df['macroarea'].isin(valid_areas)]

    if len(df) < 50 or df['family'].nunique() < 10:
        return None

    try:
        # C() for categorical macroarea
        model = smf.mixedlm(
            "feature ~ vfinal + abs_lat + C(macroarea)",
            data=df,
            groups=df["family"],
        )
        result = model.fit(reml=True, method='lbfgs')

        coef = result.params.get('vfinal', None)
        se = result.bse.get('vfinal', None)
        z = result.tvalues.get('vfinal', None)
        p = result.pvalues.get('vfinal', None)

        if coef is None:
            return None

        return {
            'coef': coef,
            'se': se,
            'z': z,
            'p': p,
            'n': len(df),
            'n_families': df['family'].nunique(),
            'n_areas': df['macroarea'].nunique(),
            'direction': '+' if coef > 0 else '-',
        }
    except Exception as e:
        print(f"    B2 model failed for {feat_id}: {e}")
        return None


# ================================================================
# B3: Family-mean aggregation
# ================================================================
def run_b3(data, gb_data, feat_id):
    """
    Aggregate to family level: mean Vfinal, mean feature rate.
    Spearman correlation on family means.
    """
    family_data = defaultdict(lambda: {'vfinals': [], 'features': []})

    for gc in data:
        if gc in gb_data and feat_id in gb_data[gc]:
            fam = data[gc]['family']
            if fam:
                family_data[fam]['vfinals'].append(data[gc]['vfinal_prop'])
                family_data[fam]['features'].append(gb_data[gc][feat_id])

    # Filter families with enough languages
    fam_means_x = []
    fam_means_y = []
    fam_names = []
    fam_sizes = []

    for fam, vals in family_data.items():
        if len(vals['vfinals']) >= MIN_LANGS_PER_FAMILY:
            mx = sum(vals['vfinals']) / len(vals['vfinals'])
            my = sum(vals['features']) / len(vals['features'])
            fam_means_x.append(mx)
            fam_means_y.append(my)
            fam_names.append(fam)
            fam_sizes.append(len(vals['vfinals']))

    if len(fam_means_x) < MIN_FAMILIES_FOR_CORR:
        return None

    # Spearman
    if HAS_SCIPY:
        rho, p = sp_stats.spearmanr(fam_means_x, fam_means_y)
    else:
        rho, p = spearman_manual(fam_means_x, fam_means_y)

    # Also Pearson
    r_p, p_p = pearson_r(fam_means_x, fam_means_y)

    return {
        'spearman_rho': rho,
        'spearman_p': p,
        'pearson_r': r_p,
        'pearson_p': p_p,
        'n_families': len(fam_means_x),
        'direction': '+' if rho > 0 else '-',
        'total_langs': sum(fam_sizes),
    }


# ================================================================
# B4: Permutation by family
# ================================================================
def run_b4(data, gb_data, feat_id):
    """
    Shuffle Vfinal within families, compute Pearson r each time.
    Return empirical p-value.
    """
    # Build data
    families = defaultdict(list)
    x_all = []
    y_all = []
    fam_indices = defaultdict(list)

    idx = 0
    for gc in data:
        if gc in gb_data and feat_id in gb_data[gc]:
            fam = data[gc]['family']
            if fam:
                x_all.append(data[gc]['vfinal_prop'])
                y_all.append(gb_data[gc][feat_id])
                fam_indices[fam].append(idx)
                idx += 1

    if len(x_all) < 50:
        return None

    # Observed correlation
    r_obs, _ = pearson_r(x_all, y_all)

    # Permutation: shuffle x within each family
    count_extreme = 0
    for _ in range(N_PERM):
        x_perm = x_all[:]
        for fam, indices in fam_indices.items():
            vals = [x_perm[i] for i in indices]
            random.shuffle(vals)
            for i, v in zip(indices, vals):
                x_perm[i] = v
        r_perm, _ = pearson_r(x_perm, y_all)
        if abs(r_perm) >= abs(r_obs):
            count_extreme += 1

    p_perm = (count_extreme + 1) / (N_PERM + 1)

    return {
        'r_obs': r_obs,
        'p_perm': p_perm,
        'n_perm': N_PERM,
        'n': len(x_all),
        'direction': '+' if r_obs > 0 else '-',
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 100)
    print("09b: GENEALOGICAL AND AREAL CONTROLS FOR PHONOTACTIC–MORPHOLOGICAL CASCADE")
    print("=" * 100)

    # Load data
    print("\nStep 1: Loading ASJP word-final V%...")
    lang_vfinal = load_asjp_wordfinal()
    print(f"  Languages with Vfinal + coords: {len(lang_vfinal)}")

    print("\nStep 2: Loading Grambank features + macroarea...")
    gb_data, gb_langs = load_grambank()
    print(f"  Languages in Grambank: {len(gb_data)}")
    print(f"  Languages with macroarea: {sum(1 for v in gb_langs.values() if v != 'Unknown')}")

    # Overlap
    overlap = set(lang_vfinal.keys()) & set(gb_data.keys())
    print(f"\n  ASJP–Grambank overlap: {len(overlap)} languages")

    # ================================================================
    # Run all controls for each feature
    # ================================================================
    all_results = {}

    for feat_id in sorted(GB_FEATURES.keys()):
        feat_name = GB_FEATURES[feat_id]
        print(f"\n{'='*80}")
        print(f"  {feat_id}: {feat_name}")
        print(f"{'='*80}")

        feat_results = {'name': feat_name}

        # --- B1: Mixed-effects with family random intercept ---
        print(f"\n  [B1] Mixed-effects LPM: feature ~ vfinal + abs_lat + (1|family)")
        b1 = run_b1(lang_vfinal, gb_data, feat_id)
        feat_results['B1'] = b1
        if b1:
            sig = '***' if b1['p'] < 0.001 else '**' if b1['p'] < 0.01 else '*' if b1['p'] < 0.05 else 'ns'
            print(f"       coef(vfinal) = {b1['coef']:+.4f}, SE = {b1['se']:.4f}, "
                  f"z = {b1['z']:.2f}, p = {b1['p']:.2e} {sig}")
            print(f"       n = {b1['n']}, families = {b1['n_families']}, direction = {b1['direction']}")
        else:
            print(f"       Skipped (insufficient data or model failure)")

        # --- B2: Macroarea fixed effect ---
        print(f"\n  [B2] + Macroarea: feature ~ vfinal + abs_lat + macroarea + (1|family)")
        b2 = run_b2(lang_vfinal, gb_data, gb_langs, feat_id)
        feat_results['B2'] = b2
        if b2:
            sig = '***' if b2['p'] < 0.001 else '**' if b2['p'] < 0.01 else '*' if b2['p'] < 0.05 else 'ns'
            print(f"       coef(vfinal) = {b2['coef']:+.4f}, SE = {b2['se']:.4f}, "
                  f"z = {b2['z']:.2f}, p = {b2['p']:.2e} {sig}")
            print(f"       n = {b2['n']}, families = {b2['n_families']}, "
                  f"areas = {b2['n_areas']}, direction = {b2['direction']}")
        else:
            print(f"       Skipped (insufficient data or model failure)")

        # --- B3: Family-mean aggregation ---
        print(f"\n  [B3] Family-mean aggregation (Spearman)")
        b3 = run_b3(lang_vfinal, gb_data, feat_id)
        feat_results['B3'] = b3
        if b3:
            sig = '***' if b3['spearman_p'] < 0.001 else '**' if b3['spearman_p'] < 0.01 else '*' if b3['spearman_p'] < 0.05 else 'ns'
            print(f"       Spearman ρ = {b3['spearman_rho']:+.4f}, p = {b3['spearman_p']:.2e} {sig}")
            print(f"       Pearson r  = {b3['pearson_r']:+.4f}, p = {b3['pearson_p']:.2e}")
            print(f"       n_families = {b3['n_families']}, total langs = {b3['total_langs']}, "
                  f"direction = {b3['direction']}")
        else:
            print(f"       Skipped (< {MIN_FAMILIES_FOR_CORR} families)")

        # --- B4: Permutation by family ---
        print(f"\n  [B4] Permutation within family ({N_PERM} iterations)")
        b4 = run_b4(lang_vfinal, gb_data, feat_id)
        feat_results['B4'] = b4
        if b4:
            sig = '***' if b4['p_perm'] < 0.001 else '**' if b4['p_perm'] < 0.01 else '*' if b4['p_perm'] < 0.05 else 'ns'
            print(f"       r_obs = {b4['r_obs']:+.4f}, p_perm = {b4['p_perm']:.4f} {sig}")
            print(f"       n = {b4['n']}, direction = {b4['direction']}")
        else:
            print(f"       Skipped (insufficient data)")

        all_results[feat_id] = feat_results

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n\n" + "=" * 120)
    print("SUMMARY: DIRECTION CONSISTENCY ACROSS ALL CONTROLS")
    print("=" * 120)

    header = (f"{'Feature':<45s} {'Raw':>5s} {'B1':>5s} {'B2':>5s} "
              f"{'B3':>5s} {'B4':>5s} {'Consistent?':>12s}")
    print(header)
    print("-" * 120)

    suffix_feats = ['GB080', 'GB089', 'GB091', 'GB093']
    prefix_feats = ['GB079', 'GB090', 'GB092', 'GB094']
    n_consistent = 0
    n_total = 0

    for feat_id in sorted(GB_FEATURES.keys()):
        res = all_results[feat_id]
        name = GB_FEATURES[feat_id]

        # Expected direction
        if feat_id in suffix_feats:
            expected = '+'
        elif feat_id in prefix_feats:
            expected = '-'
        else:
            expected = '?'

        # Collect directions
        dirs = {}

        # Raw (B4 gives us the raw r)
        if res.get('B4') and res['B4']:
            dirs['raw'] = res['B4']['direction']
        else:
            dirs['raw'] = '?'

        for key in ['B1', 'B2', 'B3', 'B4']:
            r = res.get(key)
            if r:
                dirs[key] = r['direction']
            else:
                dirs[key] = '—'

        # Check consistency
        observed_dirs = [d for d in dirs.values() if d in ('+', '-')]
        if observed_dirs:
            all_same = all(d == observed_dirs[0] for d in observed_dirs)
            consistent = '✓ YES' if all_same else '✗ NO'
            if all_same:
                n_consistent += 1
        else:
            consistent = '—'
        n_total += 1

        print(f"  {name:<43s} {dirs['raw']:>5s} {dirs.get('B1','—'):>5s} "
              f"{dirs.get('B2','—'):>5s} {dirs.get('B3','—'):>5s} "
              f"{dirs.get('B4','—'):>5s} {consistent:>12s}")

    print(f"\n  Direction-consistent features: {n_consistent}/{n_total}")

    # ================================================================
    # PREFIX–SUFFIX MIRROR TEST
    # ================================================================
    print("\n\n" + "=" * 100)
    print("PREFIX–SUFFIX MIRROR: Sign consistency in B1 (mixed-effects)")
    print("=" * 100)

    print(f"\n  {'Feature':<45s} {'Expected':>8s} {'B1 coef':>10s} {'B1 dir':>8s} {'Match':>8s}")
    print("  " + "-" * 85)

    mirror_match = 0
    mirror_total = 0
    for feat_id in suffix_feats + prefix_feats:
        name = GB_FEATURES[feat_id]
        expected = '+' if feat_id in suffix_feats else '-'
        res = all_results.get(feat_id, {})
        b1 = res.get('B1')
        if b1:
            observed = b1['direction']
            match = '✓' if observed == expected else '✗'
            if observed == expected:
                mirror_match += 1
            mirror_total += 1
            print(f"  {name:<45s} {expected:>8s} {b1['coef']:>+10.4f} {observed:>8s} {match:>8s}")
        else:
            print(f"  {name:<45s} {expected:>8s} {'—':>10s} {'—':>8s} {'—':>8s}")

    if mirror_total > 0:
        p_binom = 0.5 ** mirror_total  # probability of all matching by chance
        print(f"\n  Mirror consistency: {mirror_match}/{mirror_total}")
        print(f"  Binomial p (all {mirror_total} matching by chance): {p_binom:.6f}")

    # ================================================================
    # SAVE RESULTS as JSON
    # ================================================================
    # Convert for JSON serialization
    def clean_for_json(obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, float):
            if obj != obj:  # NaN
                return None
            return round(obj, 6)
        return obj

    output = clean_for_json(all_results)
    with open('cascade_controls_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to cascade_controls_results.json")


if __name__ == '__main__':
    main()
