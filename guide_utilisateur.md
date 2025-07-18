## Guide d’utilisateur

### 1. Prérequis

- Python 3.8+
- pip (gestionnaire de paquets)
- (Optionnel) GPU compatible CUDA pour accélérer l’indexation et l’inférence NLP
- Installation de Streamlit, FAISS, Sentence Transformers, Pandas, NumPy, Plotly, Matplotlib, NLTK

#### Installation des dépendances
```bash
pip install -r requirements.txt
# ou installation manuelle si le fichier n’existe pas
pip install streamlit faiss-cpu sentence-transformers pandas numpy plotly matplotlib nltk
```

### 2. Préparation des données

- Téléchargez les fichiers CSV des métadonnées ArXiv (liens ou exemples dans le dossier data/)
- Placez les fichiers dans le dossier adéquat (ex: ./data)

### 3. Initialisation de la base de données

Lancez le script d’initialisation pour importer les données et créer la base SQLite :
```bash
python scripts/init_db.py --input ./data/metadata.csv --output ./data/arxiv.db
```

### 4. Génération des embeddings et indexation FAISS

Lancez le pipeline pour générer les embeddings sémantiques et l’index FAISS :
```bash
python scripts/create_embeddings.py --db ./data/arxiv.db --output ./data/arxiv.index
```

### 5. Lancement de l’application Streamlit

Démarrez le chatbot en mode interface web :
```bash
streamlit run app.py
```
Accédez à l’URL indiquée (par défaut http://localhost:8501).

---

### 6. Utilisation du chatbot

- **Recherche sémantique** : Entrez une question ou un sujet scientifique pour obtenir des articles pertinents.
- **Filtres dynamiques** : Affinez votre recherche par année, catégorie, type de publication (barre latérale).
- **Visualisations** : Consultez les tendances, collaborations et sujets sous forme de graphiques interactifs.
- **Conversation** : Dialoguez avec le chatbot pour des recommandations personnalisées ou des analyses avancées.

---

### 7. Fonctionnalités avancées

- **Recherche contextuelle** : Le chatbot comprend le contexte de la conversation et adapte ses réponses.
- **Extraction d’entités** : Identifie automatiquement auteurs, domaines, institutions, etc.
- **Classification d’intentions** : Catégorise votre requête (demande d’info, recommandation, analyse…).
- **Multimodalité** : Combine texte, tableaux et graphiques dans les réponses.

---

### 8. Personnalisation

- **Modifier le pipeline** : Adaptez les scripts pour d’autres sources ou modèles NLP.
- **Changer l’interface** : Personnalisez la page Streamlit (logo, couleurs, titres).
- **Ajouter des fonctionnalités** : Étendez le chatbot (API REST, support multilingue…).

---

### 9. Dépannage courant

- **Erreur de package** : Vérifiez que tous les modules sont installés (voir logs).
- **Problème d’encodage** : Assurez-vous que vos fichiers sont en UTF-8.
- **Données manquantes** : Vérifiez le chemin et la validité des fichiers CSV.
- **Performance lente** : Utilisez un GPU ou réduisez le nombre d’articles lors des tests.

---

### 10. Ressources complémentaires

- [Documentation Streamlit](https://docs.streamlit.io/)
- [Guide Sentence Transformers](https://www.sbert.net/docs/)
- [Documentation FAISS](https://github.com/facebookresearch/faiss)
- [ArXiv API](https://arxiv.org/help/api/)
- [Exemples de données ArXiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)

---

### 11. Support

Pour toute question, ouvrez une issue sur le dépôt ou contactez les auteurs :
- Yassine OUJAMA
- Yassir M’SAAD

---
