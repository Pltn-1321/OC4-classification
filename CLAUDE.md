# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an employee attrition analysis project for TechNova, focusing on predicting employee turnover using machine learning classification models. The project analyzes data from three sources (SIRH/HR data, performance evaluations, and internal surveys) to identify key factors explaining departures and build predictive models.

## Environment Setup

This project uses **uv** as the Python package manager. Python 3.11+ is required.

### Installing Dependencies
```bash
uv sync
```

### Running Jupyter Notebooks
```bash
jupyter notebook
# or
jupyter lab
```

## Data Sources

The project integrates three CSV files located in `data/`:
- `sirh.csv` - HR system data (demographics, job function, salary, tenure)
- `eval.csv` - Performance evaluation data (annual ratings, satisfaction)
- `sondage.csv` - Internal survey data (employee well-being, includes the target variable for attrition)

All files are merged using `employee_id` as the key to create a consolidated dataset.

## Project Workflow

The analysis follows a three-phase approach structured in Jupyter notebooks:

1. **Exploration and Data Cleaning** (`01_exploration.ipynb`)
   - Load and inspect all three data sources
   - Normalize column names, handle missing values
   - Merge files into central DataFrame using `employee_id`
   - Generate descriptive statistics comparing employees who left vs stayed
   - The consolidated dataset is saved as `notebooks/dataset_final_jointures.csv`

2. **Feature Engineering** (`02_feature_engineering.ipynb` - to be created)
   - Define target variable `y` (left company = 1, stayed = 0)
   - Encode categorical variables (OneHotEncoder, Target Encoding)
   - Normalize continuous variables (salary, age, tenure)
   - Check correlations to eliminate redundancies
   - Output: prepared `X` (features) and `y` (target) for ML

3. **Modeling and Evaluation** (`03_modelisation.ipynb` - to be created)
   - Train three models: DummyClassifier (baseline), LogisticRegression, RandomForestClassifier
   - Evaluate using confusion matrix, accuracy, precision, recall, F1-score, ROC-AUC
   - Use SHAP for model interpretability

## Key Libraries

- **Data manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn
- **Model Explainability**: shap (to be added to dependencies if needed)
- **Notebooks**: jupyter, ipykernel

## Project Goals

1. Identify key factors explaining employee departures through EDA
2. Prepare clean ML-ready dataset
3. Train and evaluate classification models to predict attrition
4. Provide actionable insights and recommendations for HR team

## Expected Deliverables

- Consolidated DataFrame merging all data sources
- Exploratory insights with statistics and visualizations
- ML-ready datasets (X, y)
- Trained and evaluated classification models
- Presentation materials for HR stakeholders
