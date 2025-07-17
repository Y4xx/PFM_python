# Rapport Technique - Chatbot ArXiv

Ce dossier contient le rapport technique complet du projet de chatbot de recherche scientifique basé sur les données ArXiv, développé dans le cadre du cours de Programmation Avancée en Python à l'Université Chouaïb Doukkali.

## Contenu du Projet

### Structure des Fichiers

```
docs/
├── rapport_technique.tex    # Document principal LaTeX
├── references.bib          # Bibliographie avec citations
├── README.md              # Ce fichier (instructions)
└── generated/             # Dossier de sortie (créé lors de la compilation)
```

### Fichiers du Rapport

- **`rapport_technique.tex`** : Document LaTeX principal contenant l'intégralité du rapport technique
- **`references.bib`** : Fichier de bibliographie BibTeX avec toutes les références citées
- **`README.md`** : Instructions de compilation et informations sur le projet

## Instructions de Compilation

### Prérequis

Pour compiler le document LaTeX, vous devez avoir installé :

1. **Distribution LaTeX complète** (recommandé) :
   - **Linux (Ubuntu/Debian)** : `sudo apt-get install texlive-full`
   - **macOS** : [MacTeX](http://www.tug.org/mactex/)
   - **Windows** : [MiKTeX](https://miktex.org/) ou [TeX Live](http://www.tug.org/texlive/)

2. **Packages LaTeX requis** (inclus dans texlive-full) :
   - babel (support français)
   - inputenc, fontenc (encodage)
   - geometry (mise en page)
   - fancyhdr (en-têtes/pieds de page)
   - listings (code source)
   - tikz (diagrammes)
   - algorithm, algorithmic (algorithmes)
   - hyperref (liens hypertexte)
   - booktabs (tableaux)

### Compilation Standard

#### Méthode 1 : Compilation simple (PDF sans bibliographie)
```bash
cd docs/
pdflatex rapport_technique.tex
```

#### Méthode 2 : Compilation complète avec bibliographie
```bash
cd docs/
pdflatex rapport_technique.tex
bibtex rapport_technique
pdflatex rapport_technique.tex
pdflatex rapport_technique.tex
```

#### Méthode 3 : Compilation automatisée avec latexmk (recommandée)
```bash
cd docs/
latexmk -pdf rapport_technique.tex
```

### Scripts de Compilation

#### Linux/macOS
```bash
#!/bin/bash
# compile_report.sh

echo "Compilation du rapport technique..."
cd docs/

# Nettoyage des fichiers temporaires
rm -f *.aux *.log *.bbl *.blg *.toc *.lof *.lot *.out

# Compilation avec bibliographie
pdflatex rapport_technique.tex
bibtex rapport_technique
pdflatex rapport_technique.tex
pdflatex rapport_technique.tex

echo "Compilation terminée. Le fichier PDF est disponible : rapport_technique.pdf"
```

#### Windows (PowerShell)
```powershell
# compile_report.ps1

Write-Host "Compilation du rapport technique..."
Set-Location docs/

# Nettoyage des fichiers temporaires
Remove-Item *.aux, *.log, *.bbl, *.blg, *.toc, *.lof, *.lot, *.out -ErrorAction SilentlyContinue

# Compilation avec bibliographie
pdflatex rapport_technique.tex
bibtex rapport_technique
pdflatex rapport_technique.tex
pdflatex rapport_technique.tex

Write-Host "Compilation terminée. Le fichier PDF est disponible : rapport_technique.pdf"
```

### Compilation avec Docker (Optionnel)

Si vous préférez utiliser Docker pour éviter l'installation locale de LaTeX :

```bash
# Créer un conteneur avec TeXLive
docker run --rm -v $(pwd):/workdir -w /workdir texlive/texlive:latest pdflatex rapport_technique.tex
docker run --rm -v $(pwd):/workdir -w /workdir texlive/texlive:latest bibtex rapport_technique
docker run --rm -v $(pwd):/workdir -w /workdir texlive/texlive:latest pdflatex rapport_technique.tex
docker run --rm -v $(pwd):/workdir -w /workdir texlive/texlive:latest pdflatex rapport_technique.tex
```

## Contenu du Rapport

### Structure du Document

Le rapport technique comprend les sections suivantes :

1. **Introduction** - Contexte, motivation et objectifs
2. **Architecture du Système** - Vue d'ensemble et flux de données
3. **Pipeline de Traitement des Données** - Extraction et préprocessing
4. **Conception de la Base de Données** - Schéma relationnel SQLite
5. **Implémentation de la Recherche Sémantique** - Sentence Transformers et FAISS
6. **Techniques d'Intelligence Artificielle** - Classification d'intentions et NLP
7. **Interface Utilisateur** - Architecture Streamlit et fonctionnalités
8. **Analyse des Performances** - Métriques et optimisations
9. **Perspectives Futures** - Améliorations et extensions possibles
10. **Évaluation et Résultats** - Métriques de performance et qualité
11. **Défis et Limitations** - Difficultés rencontrées et limitations
12. **Conclusion** - Contributions et apprentissages
13. **Annexes** - Code source et détails techniques

### Technologies Documentées

Le rapport couvre l'utilisation de ces technologies principales :

- **Python** : Langage principal du projet
- **Streamlit** : Framework d'interface utilisateur web
- **Sentence Transformers** : Modèles de langue pour embeddings sémantiques
- **FAISS** : Bibliothèque de recherche de similarité vectorielle
- **SQLite** : Base de données relationnelle
- **Pandas/NumPy** : Traitement et analyse de données
- **Plotly/Matplotlib** : Visualisations interactives
- **NLTK** : Traitement du langage naturel

### Fonctionnalités Décrites

- Recherche sémantique avancée avec embeddings
- Classification automatique d'intentions utilisateur
- Extraction d'entités contextuelles
- Génération de réponses intelligentes multimodales
- Visualisations dynamiques (tendances, collaborations, topics)
- Optimisations de performance avec cache intelligent
- Architecture modulaire et scalable

## Personnalisation

### Modification du Contenu

Pour personnaliser le rapport :

1. **Informations universitaires** : Modifiez la page de titre dans `rapport_technique.tex`
2. **Auteurs** : Mettez à jour les noms des étudiants et encadrant
3. **Contenu technique** : Adaptez les sections selon votre implémentation
4. **Bibliographie** : Ajoutez vos références dans `references.bib`

### Style et Formatage

Le document utilise :
- **Police** : Computer Modern (défaut LaTeX)
- **Taille** : 12pt
- **Format** : A4
- **Marges** : 2.5cm de chaque côté
- **Interligne** : Simple
- **Couleurs** : Code syntax highlighting activé

### Ajout de Figures

Pour ajouter des figures/diagrammes :

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{nom_de_votre_figure.png}
    \caption{Légende de votre figure}
    \label{fig:votre_label}
\end{figure}
```

## Résolution de Problèmes

### Erreurs Communes

1. **Package manquant** :
   ```
   ! LaTeX Error: File `package.sty' not found.
   ```
   **Solution** : Installer le package manquant avec votre gestionnaire TeX

2. **Erreur d'encodage** :
   ```
   ! Package inputenc Error: Unicode char not set up for use with LaTeX.
   ```
   **Solution** : Vérifier l'encodage UTF-8 du fichier

3. **Références non résolues** :
   ```
   LaTeX Warning: There were undefined references.
   ```
   **Solution** : Compiler avec bibtex puis relancer pdflatex deux fois

### Optimisation de la Compilation

- Utilisez `latexmk` pour la compilation automatisée
- Nettoyez les fichiers temporaires régulièrement
- Compilez en mode draft pour les révisions rapides : `\documentclass[draft]{article}`

## Contributions

Ce rapport a été développé par :
- **Yassine OUJAMA**
- **Yassir M'SAAD**

Dans le cadre du cours de **Programmation Avancée en Python** à l'**Université Chouaïb Doukkali**, Faculté des Sciences, Département d'Informatique, El Jadida.

## Licence

Ce document est distribué sous licence académique pour l'Université Chouaïb Doukkali. Utilisation autorisée à des fins éducatives et de recherche.

---

**Note** : Ce rapport technique documente un système réel de chatbot de recherche scientifique. Toutes les métriques de performance et les exemples de code présentés sont basés sur l'implémentation effective du projet.