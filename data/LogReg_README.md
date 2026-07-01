# Logistic Regression Classification Across Four Datasets

## Overview
This project applies logistic regression to four classification problems of increasing complexity: a simple candy classification task, Titanic survival prediction, breast cancer tumor classification, and diabetes prediction. Each dataset highlights a different real-world challenge — missing data, correlated features, class imbalance, and biologically invalid data values.

## Biological / Practical Questions
- Can tumor measurements distinguish malignant from benign breast masses?
- Can patient medical measurements predict diabetes, despite class imbalance and missing data disguised as zeros?
- (Plus two non-biological benchmarks — candy composition and Titanic survival — used to build up modeling technique before tackling the medical datasets.)

## Datasets
| Dataset | Source | Challenge |
|---|---|---|
| Candy Rankings | FiveThirtyEight (GitHub CSV) | None — introductory example |
| Titanic | seaborn built-in dataset | Missing data, categorical encoding, feature engineering |
| Breast Cancer Wisconsin | scikit-learn built-in dataset | Highly correlated continuous features |
| Pima Indians Diabetes | OpenML | Class imbalance, biologically invalid zero values |

All datasets load automatically within the notebook — no manual downloads required for the Titanic, breast cancer, or diabetes sections (sklearn/seaborn/OpenML built-ins). The candy dataset downloads via `wget` from a public GitHub CSV.

## Methods & Tools

**Language:** Python 3.10+

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Logistic Regression, KNN, Decision Tree, Random Forest, preprocessing, evaluation |

**Techniques applied:**
- Feature scaling with `StandardScaler` and `make_pipeline`
- Group-wise median imputation for missing values (Titanic)
- One-hot encoding for categorical variables
- Detection and correction of biologically impossible zero values (diabetes glucose, blood pressure, BMI)
- Class imbalance handling via `class_weight='balanced'`
- Multi-model comparison (Logistic Regression, KNN, Decision Tree, Random Forest)

## Key Results
- Titanic: sex and passenger class are the strongest predictors of survival, consistent with historical evacuation patterns
- Breast Cancer: multiple models achieve strong separation between malignant and benign tumors; recall on the malignant class is prioritized over raw accuracy given the cost of false negatives in a screening context
- Diabetes: correcting invalid zero values (treating them as missing rather than literal) improves data quality before modeling; class-weighted logistic regression trades a small amount of accuracy for meaningfully better recall on diabetic cases

## How to Run

**Option 1 — Google Colab (recommended)**
1. Upload `Logistic_Regression_Classification.ipynb` to Google Colab
2. Run all cells top to bottom

**Option 2 — Local environment**
```bash
git clone https://github.com/yourusername/logistic-regression-classification
cd logistic-regression-classification
pip install -r requirements.txt
jupyter notebook Logistic_Regression_Classification.ipynb
```

## Project Structure
```
logistic-regression-classification/
├── README.md
├── Logistic_Regression_Classification.ipynb
└── requirements.txt
```
