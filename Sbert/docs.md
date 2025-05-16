# 🤖 SBERT - Appariement Sémantique

notre projet utilise **Sentence-BERT (SBERT)** pour calculer des scores de similarité sémantique entre des **offres d’emploi** et des **CVs**. SBERT fournit des **représentations vectorielles contextuelles** idéales pour des tâches d’appariement sémantique comme l’alignement CV ↔ Offre.

---

## 🧠 Pourquoi SBERT ?

Contrairement aux modèles BERT classiques qui ne sont pas conçus pour des comparaisons de phrases entières, SBERT ajoute une opération de pooling pour générer un **vecteur fixe** pour chaque phrase. Ces vecteurs peuvent ensuite être comparés via la **similarité cosinus**.

---

## 🔍 SBERT pré-entraîné pour l’appariement sémantique

Dans un premier temps, nous utilisons un modèle **pré-entraîné SBERT** (ex. `all-MiniLM-L6-v2`) pour évaluer rapidement la similarité entre un CV et une offre.

- Évaluation rapide de la proximité sémantique.
- Utilisable directement sans phase d’apprentissage.

🧪 Exemple dans [`sbert.ipynb`](./sbert.ipynb)

---

## 🔧 Ajustement (Fine-Tuning) de SBERT sur un jeu de données spécifique

Pour mieux adapter SBERT à notre cas (CV ↔ Offre), nous avons **affiné le modèle** sur un **jeu de données personnalisé** constitué de paires CV/Offre annotées manuellement avec un score de similarité.

- **Génération du Dataset** :  
  Nous avons construit ce dataset avec un générateur personnalisé disponible ici :  
  👉 [job-cv-dataset-builder](https://github.com/pfe2025project/job-cv-dataset-builder)

- **Phase de Fine-tuning** :  
  Le processus est documenté dans :  
  📄 [`sbert_finetune_cv_job.ipynb`](./sbert_finetune_cv_job.ipynb)

Cela permet au modèle de mieux comprendre le vocabulaire du domaine.

---

## 🧰 Classe utilitaire `SBERTMatching`

Nous avons créé une classe `SBERTMatching` qui encapsule la logique de chargement du modèle et de calcul de la similarité.

```python
from sbert_matcher import SBERTMatching

# Charger le modèle fine-tuné
sbert_matcher = SBERTMatching(model_path='output/sbert_regression_finetuned')

# Comparer un CV avec une offre
cv = "Ingénieur ML expérimenté avec compétences en Python et deep learning."
offre = "Recherche un ingénieur deep learning ayant de l'expérience en Python."

similarite = sbert_matcher.compute_similarity(cv, offre)
print(f"Similarité cosinus : {similarite:.4f}")
```

### 📊 Application à un DataFrame :


```python 
import pandas as pd

# Le DataFrame doit contenir les colonnes 'cv_text' et 'offer_text'
df['predicted_score'] = df.apply(
    lambda row: sbert_matcher.compute_similarity(row['cv_text'], row['offer_text']),
    axis=1
)
```


Rédigé par : [Mohamed OUABBI](https://github.com/mouabbi)

--- 