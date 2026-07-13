"""
Lexibank External Validation
=============================
Replicate key ASJP findings using Lexibank (independent dataset, IPA transcription)
to confirm results are not artifacts of ASJP transcription system.

Lexibank: 5,478 languages, 1.7M forms, IPA segments
ASJP:     11,540 languages, 566K forms, ASJPcode

If patterns replicate → robust finding, not transcription artifact
"""

import csv
from collections import Counter, defaultdict
from math import sqrt
import re

# ===== IPA FEATURE MAPPING =====
# Map IPA segments to articulatory features

IPA_VOWELS = {
    # Close
    'i': {'openness': 1, 'front': 1}, 'y': {'openness': 1, 'front': 1},
    'ɨ': {'openness': 1, 'front': 0.5}, 'ʉ': {'openness': 1, 'front': 0.5},
    'ɯ': {'openness': 1, 'front': 0}, 'u': {'openness': 1, 'front': 0},
    'ɪ': {'openness': 1.5, 'front': 1}, 'ʊ': {'openness': 1.5, 'front': 0},
    # Close-mid
    'e': {'openness': 2, 'front': 1}, 'ø': {'openness': 2, 'front': 1},
    'ɘ': {'openness': 2, 'front': 0.5}, 'ɵ': {'openness': 2, 'front': 0.5},
    'ɤ': {'openness': 2, 'front': 0}, 'o': {'openness': 2, 'front': 0},
    # Mid
    'ə': {'openness': 2.5, 'front': 0.5}, 'ɐ': {'openness': 3, 'front': 0.5},
    # Open-mid
    'ɛ': {'openness': 3, 'front': 1}, 'œ': {'openness': 3, 'front': 1},
    'ɜ': {'openness': 3, 'front': 0.5}, 'ɞ': {'openness': 3, 'front': 0.5},
    'ʌ': {'openness': 3, 'front': 0}, 'ɔ': {'openness': 3, 'front': 0},
    # Open
    'æ': {'openness': 3.5, 'front': 1}, 'a': {'openness': 4, 'front': 0.5},
    'ɶ': {'openness': 4, 'front': 1}, 'ɑ': {'openness': 4, 'front': 0},
    'ɒ': {'openness': 4, 'front': 0},
}

IPA_CONSONANTS = {
    # Labial stops
    'p': {'place': 1, 'manner': 'stop', 'voiced': 0, 'sonority': 1},
    'b': {'place': 1, 'manner': 'stop', 'voiced': 1, 'sonority': 2},
    'ɓ': {'place': 1, 'manner': 'stop', 'voiced': 1, 'sonority': 2},
    # Labial fricatives
    'f': {'place': 1.2, 'manner': 'fricative', 'voiced': 0, 'sonority': 3},
    'v': {'place': 1.2, 'manner': 'fricative', 'voiced': 1, 'sonority': 4},
    'ɸ': {'place': 1, 'manner': 'fricative', 'voiced': 0, 'sonority': 3},
    'β': {'place': 1, 'manner': 'fricative', 'voiced': 1, 'sonority': 4},
    # Labial nasals
    'm': {'place': 1, 'manner': 'nasal', 'voiced': 1, 'sonority': 6},
    'ɱ': {'place': 1.2, 'manner': 'nasal', 'voiced': 1, 'sonority': 6},
    # Labial approximant
    'w': {'place': 1, 'manner': 'approximant', 'voiced': 1, 'sonority': 7},
    'ʋ': {'place': 1.2, 'manner': 'approximant', 'voiced': 1, 'sonority': 7},
    # Dental/Alveolar stops
    't': {'place': 2.5, 'manner': 'stop', 'voiced': 0, 'sonority': 1},
    'd': {'place': 2.5, 'manner': 'stop', 'voiced': 1, 'sonority': 2},
    'ɗ': {'place': 2.5, 'manner': 'stop', 'voiced': 1, 'sonority': 2},
    'ʈ': {'place': 3, 'manner': 'stop', 'voiced': 0, 'sonority': 1},
    'ɖ': {'place': 3, 'manner': 'stop', 'voiced': 1, 'sonority': 2},
    't̪': {'place': 2, 'manner': 'stop', 'voiced': 0, 'sonority': 1},
    'd̪': {'place': 2, 'manner': 'stop', 'voiced': 1, 'sonority': 2},
    # Alveolar fricatives
    's': {'place': 2.5, 'manner': 'fricative', 'voiced': 0, 'sonority': 3},
    'z': {'place': 2.5, 'manner': 'fricative', 'voiced': 1, 'sonority': 4},
    'ʃ': {'place': 3.2, 'manner': 'fricative', 'voiced': 0, 'sonority': 3},
    'ʒ': {'place': 3.2, 'manner': 'fricative', 'voiced': 1, 'sonority': 4},
    'ʂ': {'place': 3, 'manner': 'fricative', 'voiced': 0, 'sonority': 3},
    'ʐ': {'place': 3, 'manner': 'fricative', 'voiced': 1, 'sonority': 4},
    'θ': {'place': 2, 'manner': 'fricative', 'voiced': 0, 'sonority': 3},
    'ð': {'place': 2, 'manner': 'fricative', 'voiced': 1, 'sonority': 4},
    # Alveolar affricates
    'ts': {'place': 2.5, 'manner': 'affricate', 'voiced': 0, 'sonority': 2},
    'dz': {'place': 2.5, 'manner': 'affricate', 'voiced': 1, 'sonority': 3},
    'tʃ': {'place': 3.2, 'manner': 'affricate', 'voiced': 0, 'sonority': 2},
    'dʒ': {'place': 3.2, 'manner': 'affricate', 'voiced': 1, 'sonority': 3},
    # Alveolar nasals
    'n': {'place': 2.5, 'manner': 'nasal', 'voiced': 1, 'sonority': 6},
    'ɳ': {'place': 3, 'manner': 'nasal', 'voiced': 1, 'sonority': 6},
    'ɲ': {'place': 3.5, 'manner': 'nasal', 'voiced': 1, 'sonority': 6},
    # Alveolar liquids
    'r': {'place': 2.5, 'manner': 'trill', 'voiced': 1, 'sonority': 7},
    'ɾ': {'place': 2.5, 'manner': 'flap', 'voiced': 1, 'sonority': 7},
    'ɽ': {'place': 3, 'manner': 'flap', 'voiced': 1, 'sonority': 7},
    'l': {'place': 2.5, 'manner': 'lateral', 'voiced': 1, 'sonority': 7},
    'ɭ': {'place': 3, 'manner': 'lateral', 'voiced': 1, 'sonority': 7},
    'ɬ': {'place': 2.5, 'manner': 'lateral_fricative', 'voiced': 0, 'sonority': 4},
    'ɮ': {'place': 2.5, 'manner': 'lateral_fricative', 'voiced': 1, 'sonority': 5},
    # Palatal
    'c': {'place': 3.5, 'manner': 'stop', 'voiced': 0, 'sonority': 1},
    'ɟ': {'place': 3.5, 'manner': 'stop', 'voiced': 1, 'sonority': 2},
    'ç': {'place': 3.5, 'manner': 'fricative', 'voiced': 0, 'sonority': 3},
    'ʝ': {'place': 3.5, 'manner': 'fricative', 'voiced': 1, 'sonority': 4},
    'j': {'place': 3.5, 'manner': 'approximant', 'voiced': 1, 'sonority': 7},
    'ʎ': {'place': 3.5, 'manner': 'lateral', 'voiced': 1, 'sonority': 7},
    # Velar
    'k': {'place': 4, 'manner': 'stop', 'voiced': 0, 'sonority': 1},
    'g': {'place': 4, 'manner': 'stop', 'voiced': 1, 'sonority': 2},
    'ɠ': {'place': 4, 'manner': 'stop', 'voiced': 1, 'sonority': 2},
    'x': {'place': 4, 'manner': 'fricative', 'voiced': 0, 'sonority': 3},
    'ɣ': {'place': 4, 'manner': 'fricative', 'voiced': 1, 'sonority': 4},
    'ŋ': {'place': 4, 'manner': 'nasal', 'voiced': 1, 'sonority': 6},
    # Uvular
    'q': {'place': 4.5, 'manner': 'stop', 'voiced': 0, 'sonority': 1},
    'ɢ': {'place': 4.5, 'manner': 'stop', 'voiced': 1, 'sonority': 2},
    'χ': {'place': 4.5, 'manner': 'fricative', 'voiced': 0, 'sonority': 3},
    'ʁ': {'place': 4.5, 'manner': 'fricative', 'voiced': 1, 'sonority': 4},
    'ɴ': {'place': 4.5, 'manner': 'nasal', 'voiced': 1, 'sonority': 6},
    # Glottal
    'h': {'place': 5, 'manner': 'fricative', 'voiced': 0, 'sonority': 3},
    'ɦ': {'place': 5, 'manner': 'fricative', 'voiced': 1, 'sonority': 4},
    'ʔ': {'place': 5, 'manner': 'stop', 'voiced': 0, 'sonority': 1},
}


def classify_segment(seg):
    """Classify an IPA segment. Returns ('vowel', features) or ('consonant', features) or None."""
    # Strip modifiers
    clean = seg.strip('ːˑ̃ʰʷʲˤˠ̥̤̰̹̜̩̯̊̆̈̽̚ʼ̻̝̞̟̠̀́̂̌̄̏̋̊ˈˌ')
    
    if clean in IPA_VOWELS:
        return 'vowel', IPA_VOWELS[clean]
    if clean in IPA_CONSONANTS:
        return 'consonant', IPA_CONSONANTS[clean]
    
    # Try first character for compound segments
    if len(clean) > 1:
        if clean[0] in IPA_CONSONANTS:
            return 'consonant', IPA_CONSONANTS[clean[0]]
        if clean[0] in IPA_VOWELS:
            return 'vowel', IPA_VOWELS[clean[0]]
    
    return None


def extract_word_features(segments_str):
    """Extract features from IPA segments string (space-separated, + for morpheme boundaries)."""
    if not segments_str or segments_str.strip() == '':
        return None
    
    segments = [s for s in segments_str.replace('+', ' ').split() if s.strip()]
    if not segments:
        return None
    
    openness_vals = []
    voiced_vals = []
    sonority_vals = []
    place_vals = []
    
    for seg in segments:
        result = classify_segment(seg)
        if result is None:
            continue
        seg_type, feats = result
        if seg_type == 'vowel':
            openness_vals.append(feats['openness'])
        elif seg_type == 'consonant':
            voiced_vals.append(feats['voiced'])
            sonority_vals.append(feats['sonority'])
            place_vals.append(feats['place'])
    
    if not openness_vals or not voiced_vals:
        return None
    
    return {
        'openness': sum(openness_vals) / len(openness_vals),
        'voiced': sum(voiced_vals) / len(voiced_vals) if voiced_vals else None,
        'sonority': sum(sonority_vals) / len(sonority_vals) if sonority_vals else None,
        'place': sum(place_vals) / len(place_vals) if place_vals else None,
    }


def welch_t(d1, d2):
    import math
    n1, n2 = len(d1), len(d2)
    if n1 < 2 or n2 < 2: return 0, 1.0
    m1, m2 = sum(d1)/n1, sum(d2)/n2
    v1 = sum((x-m1)**2 for x in d1)/(n1-1)
    v2 = sum((x-m2)**2 for x in d2)/(n2-1)
    se = sqrt(v1/n1 + v2/n2) if (v1/n1 + v2/n2) > 0 else 1e-15
    t = (m1-m2)/se
    p = 2*(1 - 0.5*(1+math.erf(abs(t)/math.sqrt(2))))
    return t, p


def cohens_d(d1, d2):
    n1, n2 = len(d1), len(d2)
    if n1 < 2 or n2 < 2: return 0
    m1, m2 = sum(d1)/n1, sum(d2)/n2
    v1 = sum((x-m1)**2 for x in d1)/(n1-1)
    v2 = sum((x-m2)**2 for x in d2)/(n2-1)
    ps = sqrt(((n1-1)*v1+(n2-1)*v2)/(n1+n2-2))
    return (m1-m2)/ps if ps > 0 else 0


def main():
    print("Loading Lexibank data...")
    
    # Load languages
    languages = {}
    with open('/content/lexibank-analysed/cldf/languages.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            lat = float(row['Latitude']) if row['Latitude'] else None
            languages[row['ID']] = {
                'family': row.get('Family', '') or row.get('Family_in_Data', ''),
                'lat': lat,
                'abs_lat': abs(lat) if lat is not None else None,
            }
    
    # Load concepts (map to Concepticon glosses)
    concepts = {}
    with open('/content/lexibank-analysed/cldf/concepts.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cgloss = row.get('Concepticon_Gloss', '')
            if cgloss:
                concepts[row['ID']] = cgloss
    
    print(f"  Languages: {len(languages):,}")
    print(f"  Concepts with Concepticon mapping: {len(concepts):,}")
    
    # Key concepts to test
    KEY_CONCEPTS = {
        'BIG', 'SMALL', 'LONG', 'TONGUE', 'NOSE', 'MOUTH', 'TOOTH',
        'BONE', 'STONE', 'WATER', 'DIE', 'KILL', 'I', 'BLOOD', 'SKIN',
        'BIRD', 'DOG', 'THOU'
    }
    
    # Load forms
    print("Loading forms (this may take a moment)...")
    concept_features = defaultdict(list)
    all_features = []
    total_forms = 0
    matched_forms = 0
    
    with open('/content/lexibank-analysed/cldf/forms.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_forms += 1
            
            cid = row.get('Parameter_ID', '')
            cgloss = concepts.get(cid, '')
            
            if not cgloss:
                continue
            
            segments = row.get('Segments', '')
            if not segments:
                continue
            
            # Skip loans
            if row.get('Loan', '') == 'TRUE' or row.get('Loan', '') == 'true':
                continue
            
            feats = extract_word_features(segments)
            if feats is None:
                continue
            
            matched_forms += 1
            concept_features[cgloss].append(feats)
            all_features.append(feats)
    
    print(f"  Total forms: {total_forms:,}")
    print(f"  Matched with features: {matched_forms:,}")
    print(f"  Unique concept glosses: {len(concept_features)}")
    
    # Global baselines
    g_open = [f['openness'] for f in all_features if f['openness'] is not None]
    g_voiced = [f['voiced'] for f in all_features if f['voiced'] is not None]
    g_son = [f['sonority'] for f in all_features if f['sonority'] is not None]
    g_place = [f['place'] for f in all_features if f['place'] is not None]
    
    gm = {
        'openness': sum(g_open)/len(g_open),
        'voiced': sum(g_voiced)/len(g_voiced),
        'sonority': sum(g_son)/len(g_son),
        'place': sum(g_place)/len(g_place),
    }
    
    print(f"\n  Global baselines (Lexibank IPA):")
    print(f"    Openness: {gm['openness']:.4f} (n={len(g_open):,})")
    print(f"    Voiced:   {gm['voiced']:.4f} (n={len(g_voiced):,})")
    print(f"    Sonority: {gm['sonority']:.4f} (n={len(g_son):,})")
    print(f"    Place:    {gm['place']:.4f} (n={len(g_place):,})")
    
    # ================================================================
    # CROSS-VALIDATION: ASJP vs Lexibank
    # ================================================================
    print("\n" + "="*100)
    print("CROSS-VALIDATION: Do ASJP findings replicate in Lexibank (IPA)?")
    print("="*100)
    
    print(f"\n{'Concept':<14s} {'Dim':<10s} {'n':>8s} {'Mean':>8s} {'Global':>8s} {'Δ':>8s} {'d':>8s} {'t':>8s} {'p':>12s} {'ASJP dir':>9s} {'Match':>6s}")
    print("-" * 110)
    
    ASJP_DIRECTIONS = {
        ('BIG', 'openness'): '+', ('BIG', 'voiced'): '+', ('BIG', 'sonority'): '-',
        ('SMALL', 'openness'): '-', ('SMALL', 'voiced'): '-', ('SMALL', 'sonority'): '-',
        ('LONG', 'openness'): '+', ('LONG', 'voiced'): '+', ('LONG', 'sonority'): '+',
        ('TONGUE', 'sonority'): '+', ('TONGUE', 'voiced'): '+',
        ('NOSE', 'openness'): '-', ('NOSE', 'sonority'): '+', ('NOSE', 'voiced'): '+',
        ('MOUTH', 'place'): '-',
        ('BONE', 'sonority'): '-', ('BONE', 'voiced'): '-',
        ('STONE', 'sonority'): '-', ('STONE', 'voiced'): '-',
        ('WATER', 'sonority'): '+', ('WATER', 'voiced'): '+',
        ('DIE', 'voiced'): '-', ('DIE', 'sonority'): '-',
        ('I', 'voiced'): '+', ('I', 'sonority'): '+',
    }
    
    total_predictions = 0
    matched_predictions = 0
    
    for concept in sorted(KEY_CONCEPTS):
        feats = concept_features.get(concept, [])
        if len(feats) < 50:
            continue
        
        for dim_name in ['openness', 'voiced', 'sonority', 'place']:
            dim_key = dim_name
            global_key = dim_name
            
            concept_vals = [f[dim_key] for f in feats if f[dim_key] is not None]
            if len(concept_vals) < 30:
                continue
            
            global_all = {'openness': g_open, 'voiced': g_voiced, 'sonority': g_son, 'place': g_place}
            
            cmean = sum(concept_vals)/len(concept_vals)
            delta = cmean - gm[global_key]
            t, p = welch_t(concept_vals, global_all[global_key])
            d = cohens_d(concept_vals, global_all[global_key])
            
            actual_dir = '+' if delta > 0 else '-'
            
            asjp_key = (concept, dim_name)
            asjp_dir = ASJP_DIRECTIONS.get(asjp_key, '')
            
            if asjp_dir:
                total_predictions += 1
                match = '✓' if actual_dir == asjp_dir else '✗'
                if actual_dir == asjp_dir:
                    matched_predictions += 1
            else:
                match = ''
            
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            
            if asjp_dir or (concept in KEY_CONCEPTS and dim_name in ['openness', 'voiced', 'sonority']):
                print(f"  {concept:<12s} {dim_name:<10s} {len(concept_vals):>8d} {cmean:>8.4f} {gm[global_key]:>8.4f} {delta:>+8.4f} {d:>+8.4f} {t:>8.2f} {p:>12.2e} {asjp_dir:>9s} {match:>6s}")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "="*100)
    print("REPLICATION SUMMARY")
    print("="*100)
    print(f"\n  ASJP predictions tested in Lexibank: {total_predictions}")
    print(f"  Direction matches: {matched_predictions}/{total_predictions}")
    print(f"  Replication rate: {matched_predictions/total_predictions*100:.1f}%")
    print(f"\n  Dataset comparison:")
    print(f"    ASJP:     11,540 languages, 566K forms, ASJPcode transcription")
    print(f"    Lexibank:  5,478 languages, {matched_forms:,} matched forms, IPA transcription")
    print(f"\n  If replication rate > 80%: findings are ROBUST across transcription systems")
    print(f"  If replication rate < 60%: findings may be ASJP-specific artifact")


if __name__ == '__main__':
    main()
