"""
THE CASCADE TEST: Phonology → Morphology → Syntax
===================================================
Correlate word-final V% (from ASJP) with:
  - Suffixation index (WALS 26A)
  - Inflectional synthesis (WALS 22A)
  - Number of cases (WALS 49A)
  - Fusion type (WALS 20A)
  - Word order (WALS 81A)
  - Grambank: verb suffixes, case marking, fixed order

If word-final V% correlates with morphological complexity:
  音韻 → 形態 → 統語 cascade is empirically demonstrated.
"""

import csv
from collections import defaultdict
from math import sqrt

# ================================================================
# STEP 1: Compute word-final V% per language from ASJP
# ================================================================

VOWELS = set('ieE3auo')

def load_asjp_wordfinal():
    """Compute word-final vowel % for each ASJP language."""
    languages = {}
    with open('/home/claude/asjp/cldf/languages.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            languages[row['ID']] = {
                'glottocode': row.get('Glottocode', ''),
                'name': row.get('Name', ''),
                'family': row.get('Family', ''),
                'lat': float(row['Latitude']) if row['Latitude'] else None,
            }
    
    lang_finals = defaultdict(lambda: {'v': 0, 'c': 0})
    
    with open('/home/claude/asjp/cldf/forms.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('Loan') == 'true':
                continue
            form = row['Form']
            phonemes = [ch for ch in form if ch in VOWELS or ch in set('pbfvmw8tdszclnrSZCjT5ykgxNqXh7L4G!')]
            if not phonemes:
                continue
            final = phonemes[-1]
            lang_id = row['Language_ID']
            if final in VOWELS:
                lang_finals[lang_id]['v'] += 1
            else:
                lang_finals[lang_id]['c'] += 1
    
    # Compute per-language V-final %
    lang_vfinal = {}
    for lang_id, counts in lang_finals.items():
        total = counts['v'] + counts['c']
        if total >= 20:  # minimum words
            glottocode = languages.get(lang_id, {}).get('glottocode', '')
            if glottocode:
                vf = counts['v'] / total * 100
                lang_vfinal[glottocode] = {
                    'vfinal_pct': vf,
                    'name': languages.get(lang_id, {}).get('name', ''),
                    'family': languages.get(lang_id, {}).get('family', ''),
                    'lat': languages.get(lang_id, {}).get('lat'),
                    'n_words': total,
                }
    
    return lang_vfinal


# ================================================================
# STEP 2: Load WALS features
# ================================================================

def load_wals():
    """Load WALS values for key features, indexed by Glottocode."""
    # Languages
    wals_langs = {}
    with open('/home/claude/wals/cldf/languages.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            gc = row.get('Glottocode', '')
            if gc:
                wals_langs[row['ID']] = gc
    
    # Values
    wals_data = defaultdict(dict)  # glottocode -> {feature_id: value}
    with open('/home/claude/wals/cldf/values.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            lang_id = row['Language_ID']
            gc = wals_langs.get(lang_id, '')
            if not gc:
                continue
            feat = row['Parameter_ID']
            code = row['Code_ID']
            wals_data[gc][feat] = code
    
    return wals_data


# ================================================================
# STEP 3: Load Grambank features
# ================================================================

def load_grambank():
    """Load Grambank values for key features, indexed by Glottocode."""
    gb_data = defaultdict(dict)
    with open('/home/claude/grambank/cldf/values.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            gc = row['Language_ID']  # Grambank uses Glottocodes as Language_ID
            feat = row['Parameter_ID']
            val = row['Value']
            if val in ('0', '1'):
                gb_data[gc][feat] = int(val)
    
    return gb_data


# ================================================================
# STEP 4: Encode features numerically
# ================================================================

def encode_wals_26A(code):
    """Suffixation: higher = more suffixing."""
    mapping = {
        '26A-1': 0,  # Little affixation
        '26A-6': 1,  # Strong prefixing
        '26A-5': 2,  # Weak prefixing
        '26A-4': 3,  # Equal
        '26A-3': 4,  # Weak suffixing
        '26A-2': 5,  # Strong suffixing
    }
    return mapping.get(code)

def encode_wals_22A(code):
    """Inflectional synthesis: higher = more synthetic."""
    mapping = {
        '22A-1': 0, '22A-2': 1, '22A-3': 2, '22A-4': 3,
        '22A-5': 4, '22A-6': 5, '22A-7': 6,
    }
    return mapping.get(code)

def encode_wals_49A(code):
    """Number of cases: higher = more cases."""
    mapping = {
        '49A-1': 0, '49A-9': 0,  # No case / borderline
        '49A-2': 2, '49A-3': 3, '49A-4': 4, '49A-5': 5,
        '49A-6': 6.5, '49A-7': 8.5, '49A-8': 10,
    }
    return mapping.get(code)

def encode_wals_81A(code):
    """Word order: encode as verb-final (1) vs not (0) for rigidity proxy."""
    # SOV languages tend to be more rigid; we can also look at "no dominant order"
    mapping = {
        '81A-1': 'SOV', '81A-2': 'SVO', '81A-3': 'VSO',
        '81A-4': 'VOS', '81A-5': 'OVS', '81A-6': 'OSV',
        '81A-7': 'NoDom',
    }
    return mapping.get(code)


# ================================================================
# STEP 5: Correlation computation
# ================================================================

def pearson_r(x, y):
    """Compute Pearson correlation coefficient and approximate p-value."""
    import math
    n = len(x)
    if n < 5:
        return 0, 1.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = sqrt(sum((xi - mx)**2 for xi in x) / (n-1))
    sy = sqrt(sum((yi - my)**2 for yi in y) / (n-1))
    if sx == 0 or sy == 0:
        return 0, 1.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n-1)
    r = cov / (sx * sy)
    # t-test for correlation
    if abs(r) >= 1:
        return r, 0.0
    t = r * sqrt((n-2) / (1 - r**2))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return r, p


def spearman_r(x, y):
    """Compute Spearman rank correlation."""
    def rank(data):
        sorted_data = sorted(enumerate(data), key=lambda t: t[1])
        ranks = [0] * len(data)
        for rank_val, (orig_idx, _) in enumerate(sorted_data, 1):
            ranks[orig_idx] = rank_val
        return ranks
    
    rx = rank(x)
    ry = rank(y)
    return pearson_r(rx, ry)


# ================================================================
# MAIN
# ================================================================

def main():
    print("="*100)
    print("THE CASCADE TEST: Phonology → Morphology → Syntax")
    print("Does word-final vowel % predict morphological and syntactic structure?")
    print("="*100)
    
    print("\nStep 1: Computing word-final V% per language from ASJP...")
    lang_vfinal = load_asjp_wordfinal()
    print(f"  Languages with V-final data: {len(lang_vfinal)}")
    
    print("\nStep 2: Loading WALS features...")
    wals_data = load_wals()
    print(f"  Languages in WALS: {len(wals_data)}")
    
    print("\nStep 3: Loading Grambank features...")
    gb_data = load_grambank()
    print(f"  Languages in Grambank: {len(gb_data)}")
    
    # Find overlapping languages
    all_glottocodes = set(lang_vfinal.keys())
    wals_overlap = all_glottocodes & set(wals_data.keys())
    gb_overlap = all_glottocodes & set(gb_data.keys())
    print(f"\n  Overlap with WALS: {len(wals_overlap)} languages")
    print(f"  Overlap with Grambank: {len(gb_overlap)} languages")
    
    # ================================================================
    # WALS CORRELATIONS
    # ================================================================
    print("\n" + "="*100)
    print("WALS CORRELATIONS: Word-final V% vs Morphological/Syntactic Features")
    print("="*100)
    
    # 26A: Suffixation
    print("\n--- WALS 26A: Prefixing vs Suffixing ---")
    x_suf, y_suf = [], []
    for gc in wals_overlap:
        code = wals_data[gc].get('26A')
        if code:
            val = encode_wals_26A(code)
            if val is not None:
                x_suf.append(lang_vfinal[gc]['vfinal_pct'])
                y_suf.append(val)
    
    if len(x_suf) >= 10:
        r, p = pearson_r(x_suf, y_suf)
        rs, ps = spearman_r(x_suf, y_suf)
        print(f"  n = {len(x_suf)} languages")
        print(f"  Pearson r = {r:+.4f}, p = {p:.2e}")
        print(f"  Spearman ρ = {rs:+.4f}, p = {ps:.2e}")
        print(f"  Direction: {'More V-final → More suffixing' if r > 0 else 'More V-final → Less suffixing'}")
        
        # Break down by category
        cat_means = defaultdict(list)
        for vf, sv in zip(x_suf, y_suf):
            labels = {0: 'Little affix', 1: 'Strong prefix', 2: 'Weak prefix', 
                     3: 'Equal', 4: 'Weak suffix', 5: 'Strong suffix'}
            cat_means[labels.get(sv, str(sv))].append(vf)
        
        print(f"\n  {'Category':<20s} {'n':>5s} {'Mean V-final %':>15s} {'SD':>8s}")
        print("  " + "-"*55)
        for cat in ['Little affix', 'Strong prefix', 'Weak prefix', 'Equal', 'Weak suffix', 'Strong suffix']:
            if cat in cat_means:
                vals = cat_means[cat]
                m = sum(vals)/len(vals)
                sd = sqrt(sum((v-m)**2 for v in vals)/(len(vals)-1)) if len(vals) > 1 else 0
                print(f"  {cat:<20s} {len(vals):>5d} {m:>14.1f}% {sd:>7.1f}%")
    
    # 22A: Inflectional synthesis
    print("\n--- WALS 22A: Inflectional Synthesis of the Verb ---")
    x_syn, y_syn = [], []
    for gc in wals_overlap:
        code = wals_data[gc].get('22A')
        if code:
            val = encode_wals_22A(code)
            if val is not None:
                x_syn.append(lang_vfinal[gc]['vfinal_pct'])
                y_syn.append(val)
    
    if len(x_syn) >= 10:
        r, p = pearson_r(x_syn, y_syn)
        rs, ps = spearman_r(x_syn, y_syn)
        print(f"  n = {len(x_syn)} languages")
        print(f"  Pearson r = {r:+.4f}, p = {p:.2e}")
        print(f"  Spearman ρ = {rs:+.4f}, p = {ps:.2e}")
        print(f"  Direction: {'More V-final → More synthetic' if r > 0 else 'More V-final → Less synthetic'}")
    
    # 49A: Number of cases
    print("\n--- WALS 49A: Number of Cases ---")
    x_cas, y_cas = [], []
    for gc in wals_overlap:
        code = wals_data[gc].get('49A')
        if code:
            val = encode_wals_49A(code)
            if val is not None:
                x_cas.append(lang_vfinal[gc]['vfinal_pct'])
                y_cas.append(val)
    
    if len(x_cas) >= 10:
        r, p = pearson_r(x_cas, y_cas)
        rs, ps = spearman_r(x_cas, y_cas)
        print(f"  n = {len(x_cas)} languages")
        print(f"  Pearson r = {r:+.4f}, p = {p:.2e}")
        print(f"  Spearman ρ = {rs:+.4f}, p = {ps:.2e}")
        print(f"  Direction: {'More V-final → More cases' if r > 0 else 'More V-final → Fewer cases'}")
    
    # 81A: Word Order vs V-final
    print("\n--- WALS 81A: Word Order ---")
    order_vfinal = defaultdict(list)
    for gc in wals_overlap:
        code = wals_data[gc].get('81A')
        if code:
            order = encode_wals_81A(code)
            if order:
                order_vfinal[order].append(lang_vfinal[gc]['vfinal_pct'])
    
    if order_vfinal:
        print(f"\n  {'Word Order':<15s} {'n':>5s} {'Mean V-final %':>15s} {'SD':>8s}")
        print("  " + "-"*48)
        for order in ['SOV', 'SVO', 'VSO', 'VOS', 'OVS', 'OSV', 'NoDom']:
            if order in order_vfinal:
                vals = order_vfinal[order]
                m = sum(vals)/len(vals)
                sd = sqrt(sum((v-m)**2 for v in vals)/(len(vals)-1)) if len(vals) > 1 else 0
                print(f"  {order:<15s} {len(vals):>5d} {m:>14.1f}% {sd:>7.1f}%")
        
        # Test: "No dominant order" vs others
        if 'NoDom' in order_vfinal and len(order_vfinal['NoDom']) >= 5:
            fixed = []
            for o in ['SOV', 'SVO', 'VSO', 'VOS', 'OVS', 'OSV']:
                fixed.extend(order_vfinal.get(o, []))
            free = order_vfinal['NoDom']
            if len(fixed) >= 5:
                m1 = sum(fixed)/len(fixed)
                m2 = sum(free)/len(free)
                r_f, p_f = pearson_r(
                    fixed + free,
                    [1]*len(fixed) + [0]*len(free)
                )
                print(f"\n  Fixed order mean: {m1:.1f}%, Free order mean: {m2:.1f}%")
                print(f"  Δ = {m1-m2:+.1f}%")
    
    # ================================================================
    # GRAMBANK CORRELATIONS
    # ================================================================
    print("\n" + "="*100)
    print("GRAMBANK CORRELATIONS: Word-final V% vs Morphological Features")
    print("="*100)
    
    gb_features = {
        'GB080': 'Verb suffixes/enclitics (non-person)',
        'GB070': 'Morphological case (non-pronominal)',
        'GB071': 'Morphological case (pronominal)',
        'GB082': 'Present tense marking on verbs',
        'GB083': 'Past tense marking on verbs',
        'GB084': 'Future tense marking on verbs',
        'GB086': 'Perfective/imperfective distinction',
        'GB089': 'S-argument suffix on verb',
        'GB091': 'A-argument suffix on verb',
        'GB093': 'P-argument suffix on verb',
        'GB079': 'Verb prefixes/proclitics (non-person)',
        'GB090': 'S-argument prefix on verb',
        'GB092': 'A-argument prefix on verb',
        'GB094': 'P-argument prefix on verb',
        'GB133': 'Verb-final order',
        'GB136': 'Fixed constituent order',
        'GB044': 'Productive plural marking on nouns',
    }
    
    print(f"\n  {'Feature':<50s} {'n':>5s} {'r':>8s} {'p':>12s} {'ρ':>8s} {'Direction':>30s}")
    print("  " + "-"*120)
    
    results = []
    for feat_id, feat_name in sorted(gb_features.items()):
        x_gb, y_gb = [], []
        for gc in gb_overlap:
            if feat_id in gb_data[gc]:
                x_gb.append(lang_vfinal[gc]['vfinal_pct'])
                y_gb.append(gb_data[gc][feat_id])
        
        if len(x_gb) >= 30:
            r, p = pearson_r(x_gb, y_gb)
            rs, ps = spearman_r(x_gb, y_gb)
            direction = 'V-final ↑ → feature present' if r > 0 else 'V-final ↑ → feature absent'
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"  {feat_name:<50s} {len(x_gb):>5d} {r:>+7.4f} {p:>12.2e} {rs:>+7.4f} {direction:>30s} {sig}")
            results.append((feat_id, feat_name, len(x_gb), r, p, rs, ps))
    
    # ================================================================
    # COMPOSITE INDICES
    # ================================================================
    print("\n" + "="*100)
    print("COMPOSITE INDICES")
    print("="*100)
    
    # Suffix density = average of GB080, GB089, GB091, GB093
    suffix_feats = ['GB080', 'GB089', 'GB091', 'GB093']
    prefix_feats = ['GB079', 'GB090', 'GB092', 'GB094']
    inflection_feats = ['GB070', 'GB082', 'GB083', 'GB084', 'GB086']
    
    for index_name, feat_list in [
        ('Suffix density (GB080/089/091/093)', suffix_feats),
        ('Prefix density (GB079/090/092/094)', prefix_feats),
        ('Inflectional richness (GB070/082/083/084/086)', inflection_feats),
    ]:
        x_comp, y_comp = [], []
        for gc in gb_overlap:
            vals = [gb_data[gc].get(f) for f in feat_list]
            vals = [v for v in vals if v is not None]
            if len(vals) >= len(feat_list) // 2 + 1:  # at least half features present
                x_comp.append(lang_vfinal[gc]['vfinal_pct'])
                y_comp.append(sum(vals) / len(vals))
        
        if len(x_comp) >= 30:
            r, p = pearson_r(x_comp, y_comp)
            rs, ps = spearman_r(x_comp, y_comp)
            print(f"\n  {index_name}")
            print(f"    n = {len(x_comp)}, Pearson r = {r:+.4f} (p = {p:.2e}), Spearman ρ = {rs:+.4f} (p = {ps:.2e})")
            if r > 0:
                print(f"    → More V-final → MORE {index_name.split('(')[0].strip()}")
            else:
                print(f"    → More V-final → LESS {index_name.split('(')[0].strip()}")
    
    # ================================================================
    # THE FULL CASCADE VISUALIZATION
    # ================================================================
    print("\n" + "="*100)
    print("THE FULL CASCADE: Latitude → V-final% → Morphology → Syntax")
    print("="*100)
    
    # Bin languages by V-final quartiles and show morphological profiles
    all_vf = [(gc, lang_vfinal[gc]['vfinal_pct']) for gc in gb_overlap if gc in gb_data]
    all_vf.sort(key=lambda x: x[1])
    
    n = len(all_vf)
    quartiles = [
        ('Q1 (lowest V-final)', all_vf[:n//4]),
        ('Q2', all_vf[n//4:n//2]),
        ('Q3', all_vf[n//2:3*n//4]),
        ('Q4 (highest V-final)', all_vf[3*n//4:]),
    ]
    
    print(f"\n  Languages binned by word-final vowel % quartiles (n={n}):")
    print(f"\n  {'Quartile':<25s} {'V-final%':>10s} {'Suffix':>8s} {'Prefix':>8s} {'Inflect':>8s} {'VFinal':>8s} {'FixOrd':>8s}")
    print("  " + "-"*82)
    
    for qname, qlangs in quartiles:
        vf_mean = sum(vf for _, vf in qlangs) / len(qlangs) if qlangs else 0
        
        feat_means = {}
        for feat_group, feat_ids in [
            ('Suffix', suffix_feats),
            ('Prefix', prefix_feats),
            ('Inflect', inflection_feats),
            ('VFinal', ['GB133']),
            ('FixOrd', ['GB136']),
        ]:
            vals = []
            for gc, _ in qlangs:
                for fid in feat_ids:
                    v = gb_data.get(gc, {}).get(fid)
                    if v is not None:
                        vals.append(v)
            feat_means[feat_group] = sum(vals)/len(vals) if vals else float('nan')
        
        print(f"  {qname:<25s} {vf_mean:>9.1f}% {feat_means['Suffix']:>7.3f} {feat_means['Prefix']:>7.3f} {feat_means['Inflect']:>7.3f} {feat_means['VFinal']:>7.3f} {feat_means['FixOrd']:>7.3f}")
    
    print("""
    Prediction: 
      Q4 (high V-final / tropical-like) → MORE suffix, MORE inflection
      Q1 (low V-final / subarctic-like)  → LESS suffix, LESS inflection, MORE fixed order
    
    If this pattern holds: the full cascade is demonstrated.
    """)


if __name__ == '__main__':
    main()
