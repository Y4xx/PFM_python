#!/bin/bash
# cleanup.sh - Clean LaTeX auxiliary files

echo "Nettoyage des fichiers auxiliaires LaTeX..."

rm -f *.aux *.log *.bbl *.blg *.toc *.lof *.lot *.out *.fls *.fdb_latexmk

echo "Nettoyage terminé."