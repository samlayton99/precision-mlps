#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
for note in 0*.tex; do
    if command -v tectonic >/dev/null 2>&1; then
        tectonic --untrusted "$note"
    elif command -v latexmk >/dev/null 2>&1; then
        latexmk -pdf -interaction=nonstopmode -halt-on-error "$note"
    elif command -v pdflatex >/dev/null 2>&1; then
        pdflatex -interaction=nonstopmode -halt-on-error "$note"
        pdflatex -interaction=nonstopmode -halt-on-error "$note"
    else
        echo 'Install Tectonic or a standard LaTeX distribution to rebuild the PDFs.' >&2
        exit 1
    fi
done
