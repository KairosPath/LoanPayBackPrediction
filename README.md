# 🏦 Loan Payback Prediction

![Version](https://img.shields.io/badge/version-1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.x-green.svg)
![License](https://img.shields.io/badge/license-Kaggle%20Competition-lightgrey.svg)

**Playground Series - Season 5, Episode 11**

A machine learning project that predicts whether a loan will be paid back based on borrower characteristics and loan details. This project achieved **top 25% ranking** in the [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s5e11).

## Results

- **Final Model Performance:** 0.9230 ROC AUC on validation set
- **Best Base Model:** CatBoost (0.9226 ROC AUC)
- **Improvement:** Meta-model ensemble improved performance by 0.0004 ROC AUC over the best individual model
- **Competition Ranking:** Top 25%

## Project Overview

This project explores various machine learning approaches to predict loan repayment. The final solution uses a **meta-model ensemble approach** that combines predictions from multiple gradient boosting models (CatBoost, LightGBM, and XGBoost) to achieve improved performance.

### Key Highlights

- ✅ Comprehensive EDA with correlation analysis and target analysis
- ✅ Extensive experimentation with feature engineering (ratio features, multiplication features)
- ✅ Testing of multiple ensemble methods (stacking, blending, meta-models)
- ✅ Hyperparameter tuning for base models
- ✅ Proper data leakage prevention using cross-validation

## Project Structure

```text
KG_Competitions/
│
├── Loan_payback.ipynb          # Main notebook with complete pipeline
├── requirements.txt            # Python dependencies
├── submission.csv              # Generated predictions (output file)
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

## Getting Started

### Prerequisites

Install required packages:

```bash
pip install -r requirements.txt
```
### Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KairosPath/LoanPayBackPrediction
   ```

2. **Download the Dataset:**
   - Go to the Competition Data Page.

   - Download train.csv and test.csv.

   - Create a folder named data/ in the project root and place the files there.

3. **Run the notebook:**
   - Open `Loan_payback.ipynb` in Jupyter Notebook or JupyterLab
   - Run all cells to generate predictions
   - Final predictions will be saved as `submission.csv`

## Methodology

### 1. Exploratory Data Analysis (EDA)

- **Correlation Analysis:** Identified relationships between numerical features and checked for multicollinearity
- **Target Analysis:** Analyzed feature distributions for repaid vs. defaulted loans using boxplots and barplots
- **Key Findings:**
  - Credit score shows the most pronounced difference between groups
  - Annual income and debt-to-income ratio exhibit meaningful differences
  - Categorical features (employment status, education level, loan grade) show clear patterns

### 2. Data Preprocessing

- **Transformations:**
  - `sqrt(annual_income)` - distribution normalization
  - `log(debt_to_income_ratio)` - distribution normalization
  - OneHot encoding for categorical features
  - StandardScaler for numerical features

- **Feature Engineering:** 
  - Tested ratio and multiplication features but found they decreased performance
  - Final model uses only basic transformations to avoid overfitting

### 3. Base Models

Tested and evaluated multiple models:

| Model | ROC AUC | Status |
|-------|---------|--------|
| **CatBoost** | 0.9226 | ✅ Included |
| **LightGBM** | 0.9222 | ✅ Included |
| **XGBoost** | 0.9203 | ✅ Included |
| Random Forest | 0.9136 | ❌ Excluded |
| Logistic Regression | 0.9089 | ❌ Excluded |

### 4. Ensemble Approach

**Meta-Model Strategy (Selected):**
- Uses out-of-fold predictions from base models to train a second-level model
- Prevents data leakage by ensuring each prediction comes from a model that didn't see that sample during training
- **Result:** Achieved 0.9230 ROC AUC, improving from best base model (0.9226)

**Meta-Learners Tested:**
- XGBoost Meta-model: 0.923031 ROC AUC (best)
- LightGBM Meta-model: 0.923016 ROC AUC
- CatBoost Meta-model: 0.922411 ROC AUC

**Rejected Approaches:**
- ❌ Stacking: Performance degraded compared to individual models
- ❌ Blending: All methods performed worse (0.9185-0.9189 ROC AUC)

## 🔍 Key Findings & Challenges

### Feature Engineering
- **Challenge:** Extensive feature engineering (ratio features, multiplication features) did not improve results
- **Finding:** New features introduced noise and overfitting, decreasing performance from 0.9222 to 0.9218-0.9220
- **Solution:** Used only basic transformations that normalize distributions

### Ensemble Methods
- **Challenge:** Stacking and blending approaches performed poorly
- **Finding:** Blending methods (0.9185-0.9189 ROC AUC) were worse than individual base models
- **Solution:** Meta-model approach proved most effective, learning optimal ways to combine predictions

### Model Selection
- **Challenge:** Weaker models (Random Forest, Logistic Regression) dragged down ensemble performance
- **Solution:** Included only the three strongest base models in the final ensemble

### Hyperparameter Tuning
- **Finding:** CatBoost works excellently "out of the box" with default parameters
- **Result:** Tuning CatBoost led to worse results (0.9208) due to overfitting
- **Decision:** Used default CatBoost parameters in final version

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries:**
  - `pandas` - Data manipulation
  - `numpy` - Numerical computations
  - `scikit-learn` - Machine learning pipeline and evaluation
  - `xgboost` - Gradient boosting
  - `lightgbm` - Fast gradient boosting
  - `catboost` - Gradient boosting with categorical feature handling
  - `matplotlib` & `seaborn` - Data visualization

## 📈 Model Performance Summary

### Base Models (Validation Set)
- CatBoost: **0.922630** ROC AUC
- LightGBM: **0.922160** ROC AUC
- XGBoost: **0.920346** ROC AUC

### Meta-Models (Validation Set)
- XGB Meta-model: **0.923031** ROC AUC ⭐
- LGBM Meta-model: **0.923016** ROC AUC
- CatBoost Meta-model: **0.922411** ROC AUC

## Notes

- The project uses proper cross-validation techniques to prevent data leakage
- All models are trained with `random_state=42` for reproducibility
- The final submission uses the XGBoost meta-model as it showed the best performance on validation set

## Contributing

This is a competition project. Feel free to explore the code and methodology!

## 📄 License

This project is part of the [Kaggle Playground Series - Season 5, Episode 11](https://www.kaggle.com/competitions/playground-series-s5e11) competition. The competition data and rules are governed by Kaggle's terms and conditions.

---

**Author:** Rostislav Sidenko  
**Competition:** [Kaggle Playground Series - Season 5, Episode 11](https://www.kaggle.com/competitions/playground-series-s5e11/leaderboard)







