# Bank Customer Churn — a Comparative Study of Classical ML Models

Predicting **customer churn** for a bank (will a client leave?) on the
[Bank Customer Churn](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
dataset (10 000 customers, 14 features, ~80/20 class imbalance). Binary classification on
tabular data — the bread-and-butter task of DS in banking/fintech.

The goal is not just "train a model", but to **climb a ladder of models from simple to complex,
understanding at each rung *why* the next one is needed**, and to make the modeling choices
(imbalance handling, metric selection, categorical encoding) explicit and justified.

## The ladder

```
Logistic Regression  →  Decision Tree  →  Random Forest  →  CatBoost
   linear baseline        non-linearity      bagging          gradient boosting
```

## Evaluation

- **ROC-AUC** — main metric; threshold-independent and robust to class imbalance.
- **precision / recall on the "churned" class** — the business cares about *catching leavers*
  (recall), so this matters more than overall accuracy.
- **Accuracy is deliberately ignored** — with an 80/20 split a model that always predicts
  "stays" scores 80% while learning nothing.
- **`stratify`** on the split preserves the 80/20 ratio in train and test.
- **`class_weight="balanced"` / `auto_class_weights`** compensates for the imbalance.

## Results

| Model | ROC-AUC | recall (churn) |
|-------|:---:|:---:|
| Logistic Regression | 0.777 | 0.70 |
| Decision Tree (depth 5) | 0.8379 | 0.76 |
| Random Forest (300 trees) | 0.8566 | 0.60 |
| CatBoost (one-hot categories) | 0.855 | 0.70 |
| **CatBoost (native `cat_features`)** | **0.8593** | **0.71** |

*(single stratified 80/20 split, `random_state=42`)*

## Key findings

1. **Each rung beats the previous one on ROC-AUC.** Non-linear ensemble models capture feature
   *interactions* a linear model cannot (e.g. "German customers with high balance churn more").

2. **CatBoost's native categorical handling beats one-hot** on the same data and the same
   hyperparameters (0.855 → 0.8593), and gives the best recall on churners. One-hot shatters a
   category into many sparse 0/1 columns (one split can only ask "is it France?"), whereas
   CatBoost replaces the category with an informative number — the mean target per category —
   using **ordered target statistics** (encoding each row from prior rows only) to avoid target
   leakage. Same lesson as the theory: target encoding > one-hot for categorical features.

3. **Age is the strongest churn driver**, followed by financial features (balance, salary, credit
   score). *Caveat:* tree `feature_importances_` is biased toward high-cardinality continuous
   features and understates one-hot-encoded categories — permutation importance / SHAP would be
   more reliable.

## What was practiced (interview-relevant)

Logistic regression (sigmoid, LogLoss) · decision trees (impurity/Gini, overfitting) ·
bagging vs boosting · gradient boosting & CatBoost (ordered target encoding) ·
class imbalance · precision/recall/ROC-AUC & the decision threshold trade-off ·
feature-importance bias.

## Tech stack

Python · pandas · scikit-learn · CatBoost

## How to run

```bash
pip install -r requirements.txt
# download Churn_Modelling.csv from the Kaggle link above, place it next to the notebook
jupyter notebook churn.ipynb   # Restart & Run All
```

## Next steps

- Hyperparameter tuning (GridSearchCV / Optuna) with cross-validation.
- Decision-threshold analysis tuned to the business goal (recall vs precision).
- Permutation importance / SHAP for trustworthy feature attribution.
