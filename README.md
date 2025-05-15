# CERN Dielectron Data Analysis

## Overview

This project demonstrates a complete data science workflow on the [CERN Dielectron Dataset](https://opendata.cern.ch/record/304) using Python and Google Colab. The analysis covers data exploration, physics-inspired feature engineering, visualization, and machine learning classification of high- vs. low-mass dielectron events.

## Dataset

- **Source:** [CERN Open Data Portal](https://opendata.cern.ch/record/304) / [Kaggle](https://www.kaggle.com/datasets/fedesoriano/cern-electron-collision-data)
- **Events:** 100,000 proton-proton collision events with two reconstructed electrons (2010 CMS Run)
- **Columns:**  
  - Run, Event, E1, px1, py1, pz1, pt1, eta1, phi1, Q1  
  - E2, px2, py2, pz2, pt2, eta2, phi2, Q2  
  - M (invariant mass of electron pair, GeV)

## Workflow

### 1. Setup & Data Loading

- Downloaded `dielectron.csv` from Kaggle or CERN.
- Loaded data in Colab using pandas.
- Verified contents using `df.head()`.

### 2. Data Exploration & Visualization

- **Invariant Mass Distribution**

  ```python
  plt.hist(df['M'], bins=100, color='skyblue', edgecolor='k')
  plt.xlabel('Invariant Mass $M$ [GeV]')
  plt.ylabel('Counts')
  plt.title('Dielectron Invariant Mass Distribution')
  plt.show()
  ```
  ## Feature Engineering
Create new features to enhance model performance, such as the transverse momentum sum and pseudorapidity difference:

```python
# Feature engineering
df['pt_sum'] = df['pt1'] + df['pt2']
df['eta_diff'] = abs(df['eta1'] - df['eta2'])

# Define features and target (binary classification: Z boson vs. background)
df['is_z'] = (df['M'].between(80, 100)).astype(int)  # Z boson mass window
features = ['pt1', 'pt2', 'eta1', 'eta2', 'pt_sum', 'eta_diff']
X = df[features]
y = df['is_z']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Model Training & Evaluation
Train and evaluate three machine learning models: Random Forest, AdaBoost, and XGBoost.

### Random Forest Classifier
```python
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"Random Forest CV Accuracy: {rf_scores.mean():.3f} (+/- {rf_scores.std() * 2:.3f})")

# Train and predict
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))
```

### AdaBoost Classifier
```python
# AdaBoost
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_scores = cross_val_score(ada_model, X, y, cv=5)
print(f"AdaBoost CV Accuracy: {ada_scores.mean():.3f} (+/- {ada_scores.std() * 2:.3f})")

# Train and predict
ada_model.fit(X_train, y_train)
ada_pred = ada_model.predict(X_test)
print("AdaBoost Classification Report:")
print(classification_report(y_test, ada_pred))
```

### XGBoost Classifier
```python
# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_scores = cross_val_score(xgb_model, X, y, cv=5)
print(f"XGBoost CV Accuracy: {xgb_scores.mean():.3f} (+/- {xgb_scores.std() * 2:.3f})")

# Train and predict
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))
```

## Model Comparison
Compared Random Forest, AdaBoost, and XGBoost using cross-validation.

**Best accuracy**: XGBoost (98.8%), Random Forest (97.5%).

### Key Results
| Result Description          | Details                              |
|-----------------------------|--------------------------------------|
| Z Boson Peak                | Clear peak in invariant mass distribution around 80-100 GeV |
| Random Forest Accuracy      | ~97.5%                              |
| XGBoost Accuracy            | ~98.8%                              |
| Most Important Features     | pt1, pt2, eta1, eta2                |

## How to Run
1. Download `dielectron.csv` from [CERN Open Data](http://opendata.cern.ch) or [Kaggle](https://www.kaggle.com).
2. Open the notebook in Google Colab or Jupyter.
3. Run all cells to reproduce the analysis.

## Requirements
- Python 3.x
- pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, plotly

## References
- [CERN Open Data Portal](http://opendata.cern.ch)
- [Kaggle Dataset](https://www.kaggle.com)

## License
**Data**: Creative Commons CC0

## Acknowledgements
- CMS Collaboration
- CERN Open Data Portal
- Kaggle community

For questions or suggestions, open an issue or contact the project maintainer.
