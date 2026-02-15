"""
Sound-Meaning Universal Mapping Analysis
=========================================
Hypothesis (Iizumi): Languages didn't "diverge from a proto-language" — 
they share a common deep layer because Homo sapiens shares the same 
articulatory apparatus and the same emotions/intentions.
Surface differences are environmental adaptations (like skin melanin).

Method: For each Swadesh concept, check if certain phonemes appear 
significantly more often than chance across ALL language families.
If "nose" has /n/ more than expected in unrelated language families,
that's evidence for a universal body→sound mapping, not inheritance.
"""

import csv
import sys
from collections import Counter, defaultdict
from math import log2, sqrt
import json

# ASJPcode phoneme classes (articulatory features)
ASJP_VOWELS = set('ieE3auo')
ASJP_CONSONANTS = set('pbfvmw8tdszCnjrlSZcjT5ykgxNqXh7L4G!')

# Articulatory feature mapping for ASJPcode consonants
ARTICULATORY = {
    # Labial (lips involved)
    'p': {'place': 'labial', 'manner': 'stop', 'voiced': False},
    'b': {'place': 'labial', 'manner': 'stop', 'voiced': True},
    'f': {'place': 'labial', 'manner': 'fricative', 'voiced': False},
    'v': {'place': 'labial', 'manner': 'fricative', 'voiced': True},
    'm': {'place': 'labial', 'manner': 'nasal', 'voiced': True},
    'w': {'place': 'labial', 'manner': 'approximant', 'voiced': True},
    # Dental/Alveolar (tongue tip)
    '8': {'place': 'dental', 'manner': 'fricative', 'voiced': False},
    't': {'place': 'alveolar', 'manner': 'stop', 'voiced': False},
    'd': {'place': 'alveolar', 'manner': 'stop', 'voiced': True},
    's': {'place': 'alveolar', 'manner': 'fricative', 'voiced': False},
    'z': {'place': 'alveolar', 'manner': 'fricative', 'voiced': True},
    'c': {'place': 'alveolar', 'manner': 'affricate', 'voiced': False},
    'n': {'place': 'alveolar', 'manner': 'nasal', 'voiced': True},
    'r': {'place': 'alveolar', 'manner': 'trill/tap', 'voiced': True},
    'l': {'place': 'alveolar', 'manner': 'lateral', 'voiced': True},
    # Postalveolar/Palatal
    'S': {'place': 'postalveolar', 'manner': 'fricative', 'voiced': False},
    'Z': {'place': 'postalveolar', 'manner': 'fricative', 'voiced': True},
    'C': {'place': 'postalveolar', 'manner': 'affricate', 'voiced': False},
    'j': {'place': 'postalveolar', 'manner': 'affricate', 'voiced': True},
    'T': {'place': 'retroflex', 'manner': 'stop', 'voiced': False},
    '5': {'place': 'palatal', 'manner': 'nasal', 'voiced': True},
    'y': {'place': 'palatal', 'manner': 'approximant', 'voiced': True},
    # Velar/Uvular/Glottal
    'k': {'place': 'velar', 'manner': 'stop', 'voiced': False},
    'g': {'place': 'velar', 'manner': 'stop', 'voiced': True},
    'x': {'place': 'velar', 'manner': 'fricative', 'voiced': False},
    'N': {'place': 'velar', 'manner': 'nasal', 'voiced': True},
    'q': {'place': 'uvular', 'manner': 'stop', 'voiced': False},
    'X': {'place': 'uvular', 'manner': 'fricative', 'voiced': False},
    'h': {'place': 'glottal', 'manner': 'fricative', 'voiced': False},
    '7': {'place': 'glottal', 'manner': 'stop', 'voiced': False},
    # Lateral fricatives / clicks
    'L': {'place': 'alveolar', 'manner': 'lateral_fricative', 'voiced': False},
    '4': {'place': 'alveolar', 'manner': 'lateral_flap', 'voiced': True},
    'G': {'place': 'velar', 'manner': 'fricative', 'voiced': True},
    '!': {'place': 'various', 'manner': 'click', 'voiced': False},
}

# Body/emotion-related concept categories
CONCEPT_CATEGORIES = {
    'body_parts': ['40_EYE', '41_NOSE', '42_MOUTH', '43_TOOTH', '44_TONGUE', 
                   '39_EAR', '38_HEAD', '48_HAND', '46_FOOT', '47_KNEE',
                   '49_BELLY', '50_NECK', '52_HEART'],
    'kinship': ['18_PERSON', '16_WOMAN', '17_MAN'],
    'basic_actions': ['54_DRINK', '55_EAT', '57_SEE', '58_HEAR', '61_DIE',
                      '60_SLEEP', '66_COME', '65_WALK'],
    'nature': ['75_WATER', '82_FIRE', '72_SUN', '73_MOON', '74_STAR',
               '77_STONE', '79_EARTH'],
    'pronouns': ['1_I', '2_THOU', '3_WE'],
    'size': ['13_BIG', '15_SMALL', '14_LONG'],
}


def load_data():
    """Load ASJP data from CLDF format."""
    print("Loading language data...")
    languages = {}
    with open('/content/asjp/cldf/languages.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            languages[row['ID']] = {
                'name': row['Name'],
                'family': row['Family'],
                'macroarea': row['Macroarea'],
                'lat': float(row['Latitude']) if row['Latitude'] else None,
                'lon': float(row['Longitude']) if row['Longitude'] else None,
            }
    
    print(f"  {len(languages)} language varieties loaded")
    
    # Count families
    families = set(l['family'] for l in languages.values() if l['family'])
    print(f"  {len(families)} language families")
    
    print("Loading concept parameters...")
    concepts = {}
    with open('/content/asjp/cldf/parameters.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            concepts[row['ID']] = {
                'name': row['Name'],
                'gloss': row['Concepticon_Gloss'],
            }
    
    print(f"  {len(concepts)} concepts")
    
    print("Loading word forms (this may take a moment)...")
    forms = []
    with open('/content/asjp/cldf/forms.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Loan'] == 'true':  # Skip loanwords
                continue
            forms.append({
                'lang_id': row['Language_ID'],
                'concept_id': row['Parameter_ID'],
                'form': row['Form'],
            })
    
    print(f"  {len(forms)} word forms loaded (excluding loanwords)")
    return languages, concepts, forms


def extract_phonemes(form):
    """Extract individual phonemes from ASJPcode form."""
    phonemes = []
    skip_next = False
    for i, ch in enumerate(form):
        if skip_next:
            skip_next = False
            continue
        if ch in ('~', '$', '"', '%'):  # Modifiers
            continue
        if ch in ASJP_VOWELS or ch in ASJP_CONSONANTS:
            phonemes.append(ch)
    return phonemes


def compute_baseline_frequencies(forms):
    """Compute overall phoneme frequency across all words."""
    total = Counter()
    for entry in forms:
        for ph in extract_phonemes(entry['form']):
            total[ph] += 1
    total_count = sum(total.values())
    return {ph: count / total_count for ph, count in total.items()}, total_count


def compute_concept_phoneme_matrix(forms, languages):
    """For each concept, compute phoneme frequencies per language family."""
    # concept_id -> family -> [phonemes in all words]
    concept_family_phonemes = defaultdict(lambda: defaultdict(list))
    concept_phonemes_all = defaultdict(list)
    
    for entry in forms:
        lang = languages.get(entry['lang_id'])
        if not lang:
            continue
        family = lang['family'] if lang['family'] else 'unknown'
        phonemes = extract_phonemes(entry['form'])
        concept_family_phonemes[entry['concept_id']][family].extend(phonemes)
        concept_phonemes_all[entry['concept_id']].extend(phonemes)
    
    return concept_family_phonemes, concept_phonemes_all


def compute_association_strength(concept_phonemes, baseline_freq, min_languages=50):
    """
    For each concept-phoneme pair, compute how much more/less frequent 
    that phoneme is compared to baseline.
    
    Returns: {concept_id: {phoneme: (observed_ratio, expected_ratio, log_ratio, n_words)}}
    """
    results = {}
    
    for concept_id, phoneme_list in concept_phonemes.items():
        if len(phoneme_list) < min_languages:
            continue
        
        concept_counter = Counter(phoneme_list)
        concept_total = sum(concept_counter.values())
        
        concept_results = {}
        for ph, count in concept_counter.items():
            observed = count / concept_total
            expected = baseline_freq.get(ph, 0.001)
            if expected > 0:
                log_ratio = log2(observed / expected)
                concept_results[ph] = {
                    'observed': round(observed, 4),
                    'expected': round(expected, 4),
                    'log_ratio': round(log_ratio, 3),
                    'count': count,
                    'total': concept_total,
                }
        
        results[concept_id] = concept_results
    
    return results


def cross_family_consistency(concept_family_phonemes, target_concept, target_phoneme, min_families=5):
    """
    Check if a phoneme's overrepresentation in a concept holds across 
    UNRELATED language families. This is the key test:
    If /n/ is overrepresented in "nose" across Indo-European, Sino-Tibetan,
    Niger-Congo, Austronesian, etc., it can't be inheritance — 
    it must be the body→sound mapping.
    """
    family_data = concept_family_phonemes.get(target_concept, {})
    
    family_ratios = {}
    for family, phonemes in family_data.items():
        if len(phonemes) < 10:  # Need enough data
            continue
        counter = Counter(phonemes)
        total = sum(counter.values())
        ratio = counter.get(target_phoneme, 0) / total
        family_ratios[family] = {
            'ratio': round(ratio, 4),
            'count': counter.get(target_phoneme, 0),
            'total': total,
        }
    
    if len(family_ratios) < min_families:
        return None
    
    # Count how many families show the phoneme
    families_with = sum(1 for f in family_ratios.values() if f['ratio'] > 0)
    families_above_mean = None
    
    ratios = [f['ratio'] for f in family_ratios.values()]
    mean_ratio = sum(ratios) / len(ratios)
    
    return {
        'n_families': len(family_ratios),
        'families_with_phoneme': families_with,
        'mean_ratio': round(mean_ratio, 4),
        'top_families': sorted(family_ratios.items(), key=lambda x: -x[1]['ratio'])[:10],
        'bottom_families': sorted(family_ratios.items(), key=lambda x: x[1]['ratio'])[:5],
    }


def find_position_effects(forms, languages, concepts):
    """
    Check if overrepresented phonemes tend to appear in specific positions
    (initial, medial, final) — this would indicate the body→sound mapping
    is even more constrained than just "presence."
    """
    # concept -> phoneme -> position counts
    position_data = defaultdict(lambda: defaultdict(lambda: Counter()))
    
    for entry in forms:
        form = entry['form']
        phonemes = extract_phonemes(form)
        if not phonemes:
            continue
        
        concept_id = entry['concept_id']
        for i, ph in enumerate(phonemes):
            if i == 0:
                pos = 'initial'
            elif i == len(phonemes) - 1:
                pos = 'final'
            else:
                pos = 'medial'
            position_data[concept_id][ph][pos] += 1
    
    return position_data


def main():
    # Load data
    languages, concepts, forms = load_data()
    
    # Step 1: Baseline phoneme frequencies
    print("\n" + "="*70)
    print("STEP 1: Global baseline phoneme frequencies")
    print("="*70)
    baseline_freq, total_phonemes = compute_baseline_frequencies(forms)
    print(f"Total phonemes analyzed: {total_phonemes:,}")
    print("\nTop 15 most frequent phonemes:")
    for ph, freq in sorted(baseline_freq.items(), key=lambda x: -x[1])[:15]:
        feat = ARTICULATORY.get(ph, {})
        place = feat.get('place', 'vowel')
        manner = feat.get('manner', 'vowel')
        print(f"  {ph:3s} ({place:15s} {manner:15s}): {freq:.4f} ({freq*100:.1f}%)")
    
    # Step 2: Concept-phoneme associations
    print("\n" + "="*70)
    print("STEP 2: Sound-meaning associations (concept × phoneme)")
    print("="*70)
    concept_family_phonemes, concept_phonemes_all = compute_concept_phoneme_matrix(forms, languages)
    associations = compute_association_strength(concept_phonemes_all, baseline_freq)
    
    # Find strongest associations
    print("\nTop sound-meaning associations (|log_ratio| > 0.3):")
    print(f"{'Concept':<20s} {'Phoneme':<8s} {'Observed':>10s} {'Expected':>10s} {'Log₂ratio':>10s} {'Articulatory':>25s}")
    print("-" * 85)
    
    all_associations = []
    for concept_id, phoneme_data in associations.items():
        concept_name = concepts.get(concept_id, {}).get('gloss', concept_id)
        for ph, data in phoneme_data.items():
            if abs(data['log_ratio']) > 0.3 and data['count'] > 100:
                feat = ARTICULATORY.get(ph, {})
                art_desc = f"{feat.get('place','vowel')}/{feat.get('manner','vowel')}"
                all_associations.append((concept_name, ph, data, art_desc))
    
    all_associations.sort(key=lambda x: -abs(x[2]['log_ratio']))
    for concept_name, ph, data, art_desc in all_associations[:40]:
        print(f"  {concept_name:<20s} {ph:<8s} {data['observed']:>10.4f} {data['expected']:>10.4f} {data['log_ratio']:>+10.3f} {art_desc:>25s}")
    
    # Step 3: Cross-family consistency check
    print("\n" + "="*70)
    print("STEP 3: Cross-family consistency (the KEY test)")
    print("="*70)
    print("If a phoneme is overrepresented for a concept across UNRELATED")
    print("families, it's NOT inheritance — it's body→sound universal mapping.\n")
    
    # Test key hypotheses
    test_cases = [
        ('41', 'n', 'NOSE × /n/'),
        ('41', 'N', 'NOSE × /N/ (velar nasal)'),
        ('40', 'i', 'EYE × /i/'),
        ('44', 'l', 'TONGUE × /l/'),
        ('43', 't', 'TOOTH × /t/'),
        ('43', 's', 'TOOTH × /s/'),
        ('75', 'a', 'WATER × /a/'),
        ('82', 'p', 'FIRE × /p/'),
        ('15', 'i', 'SMALL × /i/'),
        ('13', 'a', 'BIG × /a/'),
        ('1',  'm', 'I × /m/'),
        ('2',  'n', 'THOU × /n/'),
        ('30', 'a', 'BLOOD × /a/'),
        ('61', 'm', 'DIE × /m/'),
        ('72', 's', 'SUN × /s/'),
        ('39', 'k', 'EAR × /k/'),
        ('42', 'm', 'MOUTH × /m/'),
        ('48', 'm', 'HAND × /m/'),
        ('74', 's', 'STAR × /s/'),
    ]
    
    for concept_id, phoneme, label in test_cases:
        result = cross_family_consistency(concept_family_phonemes, concept_id, phoneme)
        if result:
            assoc = associations.get(concept_id, {}).get(phoneme, {})
            lr = assoc.get('log_ratio', 'N/A')
            print(f"\n  [{label}]")
            print(f"    Log₂ ratio: {lr}")
            print(f"    Families tested: {result['n_families']}")
            print(f"    Families with /{phoneme}/: {result['families_with_phoneme']} ({result['families_with_phoneme']/result['n_families']*100:.0f}%)")
            print(f"    Mean ratio across families: {result['mean_ratio']}")
            print(f"    Top families:")
            for fam, fdata in result['top_families'][:5]:
                print(f"      {fam:30s}: {fdata['ratio']:.4f} ({fdata['count']}/{fdata['total']})")
    
    # Step 4: Position analysis for strong associations
    print("\n" + "="*70)
    print("STEP 4: Position effects (initial/medial/final)")
    print("="*70)
    position_data = find_position_effects(forms, languages, concepts)
    
    position_tests = [
        ('41', 'n', 'NOSE × /n/'),
        ('44', 'l', 'TONGUE × /l/'),
        ('43', 't', 'TOOTH × /t/'),
        ('40', 'i', 'EYE × /i/'),
        ('15', 'i', 'SMALL × /i/'),
    ]
    
    for concept_id, phoneme, label in position_tests:
        pos = position_data.get(concept_id, {}).get(phoneme, {})
        if pos:
            total = sum(pos.values())
            print(f"\n  [{label}] (n={total})")
            for position in ['initial', 'medial', 'final']:
                count = pos.get(position, 0)
                pct = count / total * 100 if total > 0 else 0
                bar = '█' * int(pct / 2)
                print(f"    {position:8s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    # Step 5: Articulatory feature patterns
    print("\n" + "="*70)
    print("STEP 5: Body-part concepts × Articulatory features")
    print("="*70)
    print("Testing: Do body parts near the lips favor labial sounds?")
    print("         Do body parts involving the tongue favor lingual sounds?\n")
    
    body_concepts = {
        '42': 'MOUTH', '44': 'TONGUE', '43': 'TOOTH',
        '41': 'NOSE', '40': 'EYE', '39': 'EAR',
        '38': 'HEAD', '48': 'HAND', '46': 'FOOT',
    }
    
    # For each body part, compute the distribution of articulatory places
    for concept_id, name in body_concepts.items():
        phonemes = concept_phonemes_all.get(concept_id, [])
        if not phonemes:
            continue
        
        consonants_only = [ph for ph in phonemes if ph in ARTICULATORY]
        if not consonants_only:
            continue
        
        place_counts = Counter()
        for ph in consonants_only:
            place_counts[ARTICULATORY[ph]['place']] += 1
        
        total_cons = sum(place_counts.values())
        print(f"  {name:10s}:", end="")
        for place in ['labial', 'alveolar', 'velar', 'glottal', 'postalveolar']:
            pct = place_counts.get(place, 0) / total_cons * 100
            print(f"  {place}={pct:.1f}%", end="")
        print()
    
    # Step 6: Latitude analysis (environmental effect)
    print("\n" + "="*70)
    print("STEP 6: Latitude effect (environmental 'melanin' equivalent)")
    print("="*70)
    print("Testing: Do phoneme inventories vary with latitude?")
    print("(If so, this is the 'UV → melanin' of language.)\n")
    
    # Group languages by latitude bands
    lat_bands = {
        'tropical (0-15°)': (0, 15),
        'subtropical (15-30°)': (15, 30),
        'temperate (30-50°)': (30, 50),
        'subarctic (50-70°)': (50, 70),
    }
    
    for band_name, (lat_min, lat_max) in lat_bands.items():
        band_phonemes = Counter()
        band_langs = 0
        for entry in forms:
            lang = languages.get(entry['lang_id'])
            if not lang or lang['lat'] is None:
                continue
            abs_lat = abs(lang['lat'])
            if lat_min <= abs_lat < lat_max:
                for ph in extract_phonemes(entry['form']):
                    band_phonemes[ph] += 1
                band_langs += 1
        
        if not band_phonemes:
            continue
        
        total = sum(band_phonemes.values())
        vowel_count = sum(band_phonemes.get(v, 0) for v in ASJP_VOWELS)
        vowel_ratio = vowel_count / total * 100
        
        # Compute mouth-opening proxy: ratio of open vowels (a, o, E) vs closed (i, u, e)
        open_vowels = sum(band_phonemes.get(v, 0) for v in 'ao3E')
        close_vowels = sum(band_phonemes.get(v, 0) for v in 'iue')
        oc_ratio = open_vowels / close_vowels if close_vowels > 0 else 0
        
        nasal_count = sum(band_phonemes.get(ph, 0) for ph in 'mnN5')
        nasal_ratio = nasal_count / total * 100
        
        print(f"  {band_name:25s}: vowel%={vowel_ratio:.1f}%, open/close={oc_ratio:.2f}, nasal%={nasal_ratio:.1f}% (n={total:,})")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
