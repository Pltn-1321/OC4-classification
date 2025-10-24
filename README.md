# 🚀 Prédiction de l'Attrition des Employés - TechNova

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table des matières
- [À propos du projet](#-à-propos-du-projet)
- [Contexte métier](#-contexte-métier)
- [Technologies utilisées](#️-technologies-utilisées)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Méthodologie](#-méthodologie)
- [Résultats](#-résultats)
- [Dashboard interactif](#-dashboard-interactif)
- [Auteur](#-auteur)

## 🎯 À propos du projet

Ce projet vise à **prédire l'attrition des employés** chez TechNova à l'aide du Machine Learning et à fournir des insights actionnables pour le département RH. En exploitant plusieurs sources de données (SIRH, évaluations de performance, sondages internes), nous construisons des modèles prédictifs capables d'identifier les employés à risque de démission.

### Objectifs principaux

1. **Analyser** les facteurs clés de l'attrition des employés
2. **Construire** des modèles de classification performants
3. **Expliquer** les prédictions avec SHAP (SHapley Additive exPlanations)
4. **Déployer** un dashboard interactif pour visualiser les résultats
5. **Fournir** des recommandations actionnables aux RH

## 📊 Contexte métier

TechNova fait face à un **taux de démission élevé** qui impacte la productivité et génère des coûts de recrutement importants. L'équipe RH souhaite :

- Identifier les **profils à risque** de démission
- Comprendre les **facteurs déterminants** de l'attrition
- Mettre en place des **actions préventives** ciblées
- Optimiser la **rétention des talents**

### Sources de données

Le projet s'appuie sur 3 sources de données complémentaires :

| Source | Contenu | Variables clés |
|--------|---------|----------------|
| **SIRH** | Données démographiques et administratives | Âge, ancienneté, salaire, département, poste |
| **Évaluations** | Performance annuelle | Notes de performance, objectifs atteints |
| **Sondages** | Bien-être et satisfaction | Satisfaction, équilibre vie pro/perso, engagement |

## 🛠️ Technologies utilisées

### Langage et environnement
- **Python 3.11+**
- **uv** - Gestionnaire de paquets et d'environnements

### Bibliothèques principales

```toml
Data Science & ML
├── pandas (2.3.2)          # Manipulation de données
├── numpy (2.3.2)           # Calcul numérique
├── scikit-learn (1.7.1)    # Machine Learning
├── xgboost (3.1.0)         # Gradient boosting
└── lightgbm (4.6.0)        # Gradient boosting léger

Visualisation
├── matplotlib (3.10.6)     # Graphiques statiques
├── seaborn (0.13.2)        # Visualisations statistiques
└── plotly (6.3.0)          # Graphiques interactifs

Explicabilité & Déploiement
├── shap (0.49.1)           # Explicabilité des modèles
├── streamlit (1.50.0)      # Dashboard web
└── explainerdashboard (0.4.7) # Dashboard ML interactif

Développement
└── jupyter (1.1.1)         # Notebooks interactifs
```

## 📦 Installation

### Prérequis
- Python 3.11 ou supérieur
- [uv](https://github.com/astral-sh/uv) (recommandé) ou pip

### Installation avec uv (recommandé)

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/OC4-classification.git
cd OC4-classification

# Installer uv si nécessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer les dépendances
uv sync
```

### Installation avec pip

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/OC4-classification.git
cd OC4-classification

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

### 1. Exploration des données

Lancer le premier notebook pour l'analyse exploratoire :

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

Ce notebook permet de :
- Charger et fusionner les données sources
- Analyser les distributions et corrélations
- Visualiser les différences entre démissionnaires et non-démissionnaires
- Identifier les variables importantes

### 2. Modélisation

Lancer le notebook de modélisation :

```bash
jupyter notebook notebooks/02_modelisation.ipynb
```

Ce notebook couvre :
- Le feature engineering et l'encodage
- L'entraînement de plusieurs modèles (Régression Logistique, Random Forest, XGBoost)
- L'évaluation et la comparaison des performances
- L'optimisation des hyperparamètres

### 3. Dashboard interactif SHAP

Lancer l'application Streamlit pour explorer les prédictions :

```bash
cd notebooks
streamlit run dashboard_shap.py
```

Le dashboard sera accessible à l'adresse : `http://localhost:8501`

## 📁 Structure du projet

```
OC4-classification/
│
├── data/                           # Données sources
│   ├── sirh.csv                    # Données SIRH
│   ├── eval.csv                    # Évaluations de performance
│   └── sondage.csv                 # Sondages internes
│
├── notebooks/                      # Notebooks Jupyter
│   ├── 01_exploration.ipynb        # Analyse exploratoire
│   ├── 02_modelisation.ipynb       # Modélisation et évaluation
│   ├── dashboard_shap.py           # Application Streamlit
│   ├── dataset_final_jointures.csv # Dataset consolidé
│   ├── log_reg_pipeline.joblib     # Modèle entraîné
│   ├── X_train.joblib              # Données d'entraînement
│   ├── X_test.joblib               # Données de test
│   ├── y_test.joblib               # Labels de test
│   └── exports/                    # Exports et visualisations
│
├── presentation/                   # Supports de présentation
│   ├── presentation.pdf
│   └── Analyse_Attrition_TechNova.pptx
│
├── pyproject.toml                  # Configuration du projet
├── uv.lock                         # Verrouillage des dépendances
├── main.py                         # Point d'entrée principal
└── README.md                       # Ce fichier
```

## 🔬 Méthodologie

### 1. Exploration et préparation des données

- **Fusion des sources** : Consolidation des 3 fichiers via `employee_id`
- **Nettoyage** : Traitement des valeurs manquantes, normalisation
- **Analyse exploratoire** : Statistiques descriptives, visualisations, tests statistiques
- **Feature engineering** : Création de nouvelles variables, encodage

### 2. Modélisation

Les modèles suivants ont été testés et comparés :

| Modèle | Type | Usage |
|--------|------|-------|
| **DummyClassifier** | Baseline | Référence de performance minimale |
| **Logistic Regression** | Linéaire | Modèle interprétable et rapide |
| **Random Forest** | Ensemble | Capture des interactions non-linéaires |
| **XGBoost / LightGBM** | Gradient Boosting | Performance maximale |

### 3. Évaluation

Métriques d'évaluation utilisées :

- **Accuracy** : Taux de prédictions correctes
- **Precision** : Qualité des prédictions positives
- **Recall (Sensibilité)** : Capacité à détecter les vrais démissionnaires
- **F1-Score** : Moyenne harmonique de Precision et Recall
- **ROC-AUC** : Capacité de discrimination du modèle
- **Matrice de confusion** : Analyse détaillée des erreurs

### 4. Explicabilité

Utilisation de **SHAP** (SHapley Additive exPlanations) pour :

- Identifier les features les plus importantes
- Expliquer chaque prédiction individuellement
- Visualiser l'impact des variables sur le risque d'attrition
- Détecter les interactions entre variables

## 📈 Résultats

### Insights clés

Les principaux facteurs d'attrition identifiés sont :

1. **Satisfaction au travail** : Principal prédicteur de l'attrition
2. **Salaire** : Impact significatif, notamment pour les postes juniors
3. **Ancienneté** : Risque élevé entre 1 et 3 ans d'ancienneté
4. **Équilibre vie pro/perso** : Facteur important de rétention
5. **Évolution de carrière** : Les employés sans progression sont plus à risque

### Performances du modèle

Les modèles atteignent des performances significativement supérieures au baseline, permettant d'identifier efficacement les employés à risque de démission.

## 📊 Dashboard interactif

Le dashboard Streamlit offre 4 onglets principaux :

### 📊 Vue d'ensemble
- **Summary Plot** : Importance et impact des features
- **Feature Importance** : Classement des variables les plus influentes

### 👤 Prédiction individuelle
- Probabilité de démission pour un employé spécifique
- **Waterfall Plot** : Décomposition de la prédiction
- **Force Plot** : Visualisation des contributions
- Top 5 des facteurs explicatifs

### 📈 Feature Importance
- Classement détaillé de toutes les variables
- Visualisation de l'importance moyenne |SHAP value|

### 🔍 Analyse de dépendance
- Relation entre une feature et son impact SHAP
- Détection d'interactions entre variables
- Effets non-linéaires

## 💡 Recommandations RH

Sur la base des analyses, voici les actions recommandées :

### Actions immédiates
- ✅ Identifier et surveiller les employés à haut risque (probabilité > 70%)
- ✅ Mener des entretiens de rétention ciblés
- ✅ Ajuster les salaires des postes sous-payés

### Actions moyen terme
- 📅 Améliorer les programmes de développement de carrière
- 📅 Renforcer l'équilibre vie pro/perso (télétravail, horaires flexibles)
- 📅 Créer des plans de rétention personnalisés pour les talents clés

### Actions long terme
- 🎯 Refondre la politique de rémunération
- 🎯 Améliorer la culture d'entreprise et l'engagement
- 🎯 Mettre en place un monitoring continu de l'attrition

## 🎓 Livrables

- ✅ **DataFrame consolidé** (3 sources fusionnées)
- ✅ **Notebooks documentés** (exploration, modélisation)
- ✅ **Modèles entraînés et sauvegardés** (joblib)
- ✅ **Dashboard interactif SHAP** (Streamlit)
- ✅ **Présentation PowerPoint/PDF** pour les RH
- ✅ **Insights et recommandations actionnables**

## 👤 Auteur

**Pierre PLUTON**

Data Scientist - Projet d'analyse de l'attrition chez TechNova

*Parcours AI Engineer - OpenClassrooms*

---

## 📄 Licence

Ce projet est réalisé dans un cadre pédagogique.

## 🙏 Remerciements

- L'équipe pédagogique OpenClassrooms
- L'équipe RH de TechNova pour les données et le contexte métier
- La communauté open-source pour les excellentes bibliothèques utilisées

---

**Note** : Ce projet est réalisé à des fins éducatives dans le cadre du parcours AI Engineer d'OpenClassrooms.
