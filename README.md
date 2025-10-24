# 🚀 Analyse de l’Attrition des Employés – TechNova

## 📌 Contexte du projet
TechNova fait face à un **taux de démission élevé**.  
L’équipe RH souhaite identifier les **facteurs clés expliquant les départs** et construire un **modèle prédictif** capable d’estimer la probabilité qu’un employé quitte l’entreprise.

Objectifs du projet :
1. Réaliser une **analyse exploratoire des données (EDA)** afin de mettre en évidence les différences entre les employés restés et ceux partis.
2. Préparer un **jeu de données propre** et prêt pour l’apprentissage automatique.
3. Entraîner et évaluer des **modèles de classification** pour prédire la démission.
4. Fournir des **insights et recommandations** actionnables pour les RH.

---

## 📂 Sources de données
Nous disposons de 3 fichiers issus de systèmes différents :
- **SIRH (RH)** → données démographiques, fonction, salaire, ancienneté, etc.
- **Évaluations de performance** → notes annuelles, satisfaction.
- **Sondage interne** → bien-être des employés, incluant un indicateur de démission (variable cible).

---

## 🛠️ Environnement technique
- **Langage** : Python 3.10+
- **Bibliothèques principales** :  
  - Manipulation : `pandas`, `numpy`  
  - Visualisation : `matplotlib`, `seaborn`, `plotly`  
  - Machine Learning : `scikit-learn`  
  - Explicabilité des modèles : `shap`  
  - Notebooks : `jupyter`  

---

## 📊 Déroulé du projet

### 1. Exploration et nettoyage des données
- Charger et inspecter les fichiers.
- Normaliser les noms de colonnes, gérer les valeurs manquantes.
- Identifier les variables quantitatives et qualitatives.
- Fusionner les fichiers dans un **DataFrame central** (clé : `employee_id`).
- Produire statistiques descriptives et visualisations comparant démissionnaires vs non-démissionnaires.

### 2. Feature Engineering
- Définir la cible `y` (a quitté l’entreprise = 1, toujours en poste = 0).
- Encoder les variables catégorielles (OneHotEncoder, Target Encoding si nécessaire).
- Normaliser les variables continues (salaire, âge, ancienneté).
- Vérifier les corrélations (Pearson, Spearman) pour éliminer les redondances.
- Résultat attendu :  
  - `X` → Features prêtes pour le ML  
  - `y` → Variable cible  

### 3. Modélisation et évaluation
- Séparer en jeux d’entraînement et de test.
- Entraîner trois modèles :
  - `DummyClassifier` (baseline)
  - `LogisticRegression` (linéaire)
  - `RandomForestClassifier` (non-linéaire)
- Évaluer avec :
  - Matrice de confusion
  - Accuracy, Précision, Rappel, F1-score
  - ROC-AUC
- Comparer avec le modèle baseline pour juger la valeur ajoutée.
- Préparer l’utilisation de **SHAP** pour interpréter les résultats.

---

## 📁 Structure du dépôt
├── data/
│ ├── sirh.csv
│ ├── eval.csv
│ └── sondage.csv
│
├── notebooks/
│ ├── 01_exploration.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_modelisation.ipynb
│
├── pyproject.toml
├── README.md
└── presentation/
    ├── presentation.pdf
    └── Analyse_Attrition_TechNova.pptx

---

## 📌 Livrables attendus
- **DataFrame central** consolidant toutes les sources.
- **Insights exploratoires** avec statistiques et graphiques.
- **Jeux de données (X, y)** prêts pour la modélisation.
- **Modèles de classification** entraînés et évalués.
- **Support de présentation** à destination des RH.

---

## 🎯 Insights attendus pour les RH
- Identifier les écarts de salaire, ancienneté, âge et satisfaction entre partants et restants.
- Repérer les profils les plus à risque.
- Fournir un outil prédictif pour estimer le risque de démission.
- Proposer des actions concrètes pour réduire l’attrition (ajustements salariaux, programmes d’engagement, etc.).

---

## 👤 Auteur
Pierre PLUTON
Projet de Data Scientist – TechNova  
*(Parcours AI Engineer – OpenClassrooms)*

