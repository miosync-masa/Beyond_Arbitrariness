# Beyond Arbitrariness: Analysis Code

**Paper:** *Beyond Arbitrariness: Universal Body–Sound Mapping Across 11,540 Language Varieties*

**Target:** Proceedings of the National Academy of Sciences (PNAS)

---

## Overview

This repository contains the complete analysis pipeline for the Beyond Arbitrariness paper.
The code demonstrates that the relationship between sound and meaning in human language
is systematically constrained by the physical properties of the human articulatory apparatus
(Thought→Body→Sound framework), and that word-final phonotactic geometry predicts
morphological strategy across languages (Universal Articulatory Substrate framework).

## Data Sources

| Database | Languages | Forms | Transcription | Source |
|----------|-----------|-------|--------------|--------|
| ASJP v21 | 11,540 | 566,435 | ASJPcode (41 symbols) | https://asjp.clld.org/ |
| Lexibank | 5,478 | 1,708,281 | IPA | https://github.com/lexibank/lexibank-analysed |
| Grambank | 2,467 | 443,779 values | Binary (0/1) | https://github.com/grambank/grambank |
| WALS | 2,501 | — | Categorical | https://github.com/cldf-datasets/wals |

### Data Setup

```bash
# Clone ASJP
git clone https://github.com/lexibank/asjp.git

# Clone Lexibank (analysed version with IPA segments)
git clone https://github.com/lexibank/lexibank-analysed.git

# Clone Grambank
git clone https://github.com/grambank/grambank.git

# Clone WALS
git clone https://github.com/cldf-datasets/wals.git
```

All scripts assume these repositories are cloned as siblings in the working directory.
Adjust paths in scripts as needed.

## Scripts (Execution Order)

### Phase 1: Core TBS Analysis (ASJP)

| # | Script | Section | What it does |
|---|--------|---------|-------------|
| 01 | `01_core_sound_meaning.py` | §4.1–4.5 | Core sound–meaning analysis: openness, voicing, sonority, place of articulation for 100 Swadesh concepts across 11,540 languages. Computes log₂ deviations and family-level significance. |
| 02 | `02_articulatory_profiles.py` | §4.2–4.7 | Multi-dimensional articulatory profiling: body-part self-naming, size–openness mapping, voicing–animacy correlation, semantic category profiles. |
| 03 | `03_correlation_matrix.py` | §4.6 | Inter-dimension correlation matrix (openness × voicing × sonority × place). Computes Nyholt effective dimensions (M_eff) for multiple testing correction. |
| 04 | `04_statistical_rigor.py` | §4.8–4.9 | FDR correction (Benjamini-Hochberg), phylogenetic family-level control (Cohen's d per family), 10,000-iteration permutation test. |
| 05 | `05_multi_dimensional_coherence.py` | §4.6, §5.2 | Multi-dimensional coherence analysis: BIG/SMALL/I symmetric structure, correlation-corrected joint p-values (Nyholt method), combined significance across dimensions. |

### Phase 2: Environmental Adaptation (ASJP)

| # | Script | Section | What it does |
|---|--------|---------|-------------|
| 06 | `06_word_final_gradient.py` | §4.10–4.11 | Latitude gradient analysis: overall vowel ratio + **word-final vowel gradient** (71.3%→41.6%, d=1.09). Position-specific analysis (word-final vs word-initial). Family-level Welch's t-test. Final consonant cluster analysis. |

### Phase 3: External Validation (Lexibank)

| # | Script | Section | What it does |
|---|--------|---------|-------------|
| 07 | `07_lexibank_sound_meaning.py` | §4.12 | Lexibank replication of sound–meaning mapping: IPA-based articulatory feature extraction (50+ vowels, 80+ consonants). Tests 25 directional predictions from ASJP → 92% replication. |
| 08 | `08_lexibank_word_final.py` | §4.12 | Lexibank replication of word-final vowel gradient: 76.6%→42.9% (d=1.49). Family-level control, consonant clusters, position-specific analysis. Independent confirmation with IPA transcription. |

### Phase 4: Phonotactic–Morphological Cascade (Grambank + WALS)

| # | Script | Section | What it does |
|---|--------|---------|-------------|
| 09 | `09_cascade_grambank_wals.py` | §4.13 | The cascade test: links ASJP word-final V% to grammatical features. WALS: suffixation (26A), synthesis (22A), cases (49A), word order (81A). Grambank: 17 morphological features. Discovers prefix↔suffix mirror pattern (p=10⁻⁸). Quartile analysis. |

## Key Results Summary

### TBS: Sound–Meaning Mapping
- 84% of predictions significant (FDR corrected)
- Joint p = 7.1 × 10⁻²⁷ for BIG (correlation-corrected)
- TONGUE × /l/: +1.546 log₂, 59% of families
- NOSE × /n/: +0.524 log₂, 69% of families
- Lexibank replication: 92% (23/25 directional predictions)

### UAS: Word-Final Vowel Gradient
- ASJP: 71.3% → 41.6% (Δ = 29.6 ppt, d = 1.09, 95% CI [0.67, 1.51])
- Lexibank: 76.6% → 42.9% (Δ = 33.7 ppt, d = 1.49, 95% CI [1.03, 1.89])
- Effect concentrates at word-final position (4× stronger than overall)
- /a/ (most open vowel) shows steepest decline

### Cascade: Phonotactics → Morphological Strategy
- NOT simplification — REALLOCATION
- High V-final → person prefixes + non-person suffixes + fewer cases
- Low V-final → person suffixes + richer case systems
- Prefix↔suffix mirror: r = ±0.08–0.12, p = 10⁻⁵ to 10⁻⁸
- 2,000+ languages × Grambank/WALS

## Dependencies

```
Python 3.8+
Standard library only (csv, math, collections, random)
No external packages required.
```

## Citation

[To be added upon publication]

---

*Differentiation is necessary. Separation is not.*

— M. Iizumi
