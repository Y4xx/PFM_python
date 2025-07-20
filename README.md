# Chatbot de Recherche Scientifique — Documentation Technique

## Présentation Générale

Ce projet propose un chatbot intelligent de recherche scientifique basé sur les données ArXiv. Son objectif principal est d’assister chercheurs et étudiants dans la découverte d’articles pertinents, l’extraction d’informations contextuelles et la visualisation avancée des tendances en recherche. Développé dans le cadre du cours de Programmation Avancée en Python à l’Université Chouaïb Doukkali, il combine les dernières avancées en NLP, recherche vectorielle et interfaces web interactives.

---

## Structure des Dossiers et Fichiers

L’organisation du projet est illustrée ci-dessous :

![image1](image1)

- **data/**  
  Contient les jeux de données ArXiv (articles et auteurs, versions nettoyées ou originales), ainsi que le dossier `data_prep` pour les notebooks de préparation, indexation et stockage.
  - `data_cleaning/data-cleaning.ipynb` : Préparation des données
  - `data_cleaning/indexing.ipynb` : Indexation vectorielle
  - `data_cleaning/storing.ipynb` : Stockage relationnel

- **docs/**  
  Documentation technique, rapport LaTeX et PDF, bibliographie, script de nettoyage.
  - `rapport_technique.tex` : Rapport technique principal (LaTeX)
  - `rapport_technique01.pdf` : Rapport technique compilé (PDF)
  - `README.md` : Guide technique de la documentation
  - `references.bib` : Bibliographie BibTeX
  - `cleanup.sh` : Script de nettoyage des fichiers temporaires

- **myenv/**  
  Environnement virtuel Python du projet (binaires, librairies, configuration).

- **racine du projet**  
  - `.gitignore` : Fichiers ignorés par Git
  - `app.py` : Application principale Streamlit
  - `article_ids.pkl` : Identifiants des articles (pickle)
  - `arxiv_database.db` : Base de données SQLite générée
  - `faiss_index.bin` : Index vectoriel FAISS
  - `requirements.txt` : Dépendances Python nécessaires
  - `runtime.txt` : Informations sur l’environnement d’exécution

---

## Objectifs du Projet

- Concevoir un assistant conversationnel capable de comprendre et de traiter des requêtes scientifiques complexes.
- Mettre en œuvre un système de recherche sémantique performant pour explorer les métadonnées ArXiv.
- Automatiser le pipeline d’extraction, de nettoyage et de normalisation des données scientifiques.
- Classifier les intentions utilisateur et extraire les entités contextuelles importantes.
- Proposer des visualisations dynamiques et des analyses bibliométriques pour faciliter la veille scientifique.

---

## Architecture du Système

Le système est structuré en plusieurs couches modulaires :

1. **Acquisition des Données**
   - Extraction des métadonnées depuis les fichiers CSV ArXiv.
   - Nettoyage et normalisation via Pandas/NumPy.
2. **Base de Données Relationnelle**
   - Stockage structuré des articles dans une base SQLite pour l’accès rapide et les jointures complexes.
3. **Pipeline NLP et Embeddings**
   - Génération de représentations vectorielles avec Sentence Transformers.
   - Indexation des embeddings dans FAISS pour une recherche sémantique efficace.
4. **Recherche et Classification**
   - Moteur de recherche sémantique : matching vectoriel, extraction d’entités, classification contextuelle.
   - Optimisation des performances et gestion du cache pour accélérer les réponses.
5. **Interface Utilisateur**
   - Application web interactive basée sur Streamlit.
   - Visualisation des résultats avec Plotly et Matplotlib.
   - Filtres dynamiques (période, domaine, type de publication).

---

## Technologies Utilisées

- **Python** : Langage principal pour le développement.
- **Streamlit** : Interface utilisateur web, expérience interactive.
- **Sentence Transformers** : Modèles NLP pour embeddings sémantiques.
- **FAISS** : Indexation et recherche vectorielle.
- **SQLite** : Stockage relationnel optimisé.
- **Pandas, NumPy** : Manipulation et analyse des données.
- **Plotly, Matplotlib** : Visualisations interactives et graphiques avancés.
- **NLTK** : Traitement du langage naturel (extraction d’entités, classification).
- **Scripts Bash et PowerShell** : Pour automatiser la compilation du rapport technique.

---

## Fonctionnalités Clés

- **Recherche sémantique avancée** : Trouver facilement les articles pertinents via embeddings et indexation vectorielle.
- **Classification automatique** : Détection et catégorisation des intentions utilisateur.
- **Extraction contextuelle** : Identification des entités et concepts clés dans les requêtes.
- **Génération de réponses multimodales** : Texte, données, tableaux, graphiques.
- **Visualisations dynamiques** : Analyse des tendances, collaborations, distribution par domaine, etc.
- **Optimisation des performances** : Système de cache intelligent, index FAISS compressé.
- **Scalabilité** : Supporte jusqu’à 1 million d’articles et 100+ utilisateurs concurrents.

---

## Analyse des Performances

- Précision@5 : **89.2%**
- Précision@10 : **85.7%**
- Rappel@20 : **92.4%**
- Score F1 moyen : **87.8%**
- Architecture stateless, support de forte concurrence et gestion mémoire optimisée.

---

## Défis Techniques & Limitations

- **Optimisation GPU** : Traitement des embeddings sur de grands volumes.
- **Gestion de la qualité des données** : Normalisation, filtrage, déduplication.
- **Limitations fonctionnelles** : 
  - Domaine restreint à l’informatique
  - Langue anglaise uniquement
  - Données statiques (mise à jour manuelle)
  - Analyse limitée au niveau des métadonnées (pas d’analyse de texte complet PDF)

---

## Perspectives d’Évolution

- **Support multilingue** : Intégration de modèles NLP multi-langues.
- **Résumé automatique** : Génération de synthèses adaptatives des articles.
- **Extraction avancée** : Reconnaissance automatique d’entités scientifiques, classification fine par sous-domaines.
- **Extensions disciplinaires** : Physique, biologie, mathématiques, etc.
- **Interopérabilité** : API pour intégration avec plateformes universitaires et systèmes externes.

---

## Impact et Applications

- Assistant de recherche pour étudiants, enseignants, laboratoires.
- Outil de veille scientifique et d’analyse bibliométrique.
- Système de recommandations personnalisées.
- Base pour des projets futurs en IA appliquée à la recherche.

---

## Auteur·es et Licence

Développé par :
- **Yassine OUJAMA**
- **Yassir M’SAAD**

Université Chouaïb Doukkali, Faculté des Sciences, Département d’Informatique, El Jadida.

Licence académique — usage éducatif et recherche exclusivement.

---

## Pour aller plus loin

Le rapport technique complet, incluant les schémas, les résultats des tests et les détails d’implémentation, est disponible dans le dossier `/docs`.

Pour compiler le rapport technique :  
Voir les instructions détaillées dans [docs/README.md](./docs/README.md).
