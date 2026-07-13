#!/usr/bin/env bash
# Compile the Supplementary Information PDF.
# Runs latexmk (preferred) or a pdflatex+bibtex+pdflatex+pdflatex fallback,
# from the supplement/ directory so that tables/ and figures/ resolve.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE" || exit 2

echo "== Compiling supplement.tex in $HERE =="
if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode supplement.tex
    rc=$?
else
    echo "latexmk not found; falling back to pdflatex/bibtex"
    pdflatex -interaction=nonstopmode supplement.tex
    bibtex supplement
    pdflatex -interaction=nonstopmode supplement.tex
    pdflatex -interaction=nonstopmode supplement.tex
    rc=$?
fi

echo
echo "== Build diagnostics =="
LOG=supplement.log
if [ -f "$LOG" ]; then
    echo "--- LaTeX errors (!) ---";            grep -n '^!' "$LOG" || echo "  none"
    echo "--- Undefined references/citations ---"; grep -n -i 'undefined' "$LOG" || echo "  none"
    echo "--- Missing figures/inputs ---";       grep -n -i 'file .* not found\|No file' "$LOG" || echo "  none"
    echo "--- Overfull boxes ---";               grep -c 'Overfull' "$LOG" | sed 's/^/  count=/'
    echo "--- Duplicate labels ---";             grep -n -i 'multiply defined\|duplicate' "$LOG" || echo "  none"
fi
if [ -f supplement.pdf ]; then
    echo "== supplement.pdf built ($(du -h supplement.pdf | cut -f1)) =="
else
    echo "== supplement.pdf NOT produced =="
fi
exit $rc
