# 🧩 Classification

> Binary classification pipelines on two classic datasets — Titanic survival and Heart Disease prediction — comparing five algorithms side by side.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-f7931e?logo=scikitlearn&logoColor=white)

---

## 📓 Notebooks

| Notebook | Dataset | Target | Rows |
|---|---|---|---|
| [`titanic_classification.ipynb`](./titanic_classification.ipynb) | Titanic (via `seaborn`) | `survived` (0/1) | 891 |
| [`heart_disease_classification.ipynb`](./heart_disease_classification.ipynb) | UCI Heart Disease (`../datasets/heart.csv`) | `HeartDisease` (0/1) | 918 |

Both notebooks follow the same core pipeline:

```
Load → Clean → Encode → Train/Test Split → Scale → Train 5 Models → Compare → Report Best Model
```

The five models trained and compared in each notebook:
- 🔵 Logistic Regression
- 🟢 K-Nearest Neighbors (KNN)
- 🟡 Naive Bayes
- 🟠 Decision Tree
- 🟣 Support Vector Machine (SVM, RBF kernel)

---

## 🚢 `titanic_classification.ipynb`

**Goal:** Predict whether a passenger survived the Titanic disaster.

**Steps:**
1. **Load** — pulls the dataset directly via `seaborn.load_dataset('titanic')`, no external file needed.
2. **Clean up** — drops redundant/duplicate columns: `deck` (>75% missing), `embark_town`, `alive`, `class`, `who`, `adult_male`. Fills missing `age` with the mean, drops the few rows missing `embarked`.
3. **Encode** — `sex` and `embarked` are label-encoded; `alone` is cast to int; all features are cast to `int` for consistency.
4. **Split & Scale** — 80/20 stratified train/test split, then `StandardScaler` (fit on train only).
5. **Train & Compare** — fits all 5 models, ranks them by **accuracy**.
6. **Report** — prints a confusion matrix and full classification report (precision/recall/F1) for the top-ranked model.

---

## ❤️ `heart_disease_classification.ipynb`

**Goal:** Predict presence of heart disease from clinical measurements.

**Steps:**
1. **Load** — reads `../datasets/heart.csv` (UCI Heart Disease dataset).
2. **Clean up** — replaces biologically impossible `0` values in `Cholesterol` and `RestingBP` with the column mean (a `0` cholesterol or blood pressure reading isn't physiologically valid — it's a data entry artifact).
3. **Encode** — one-hot encodes all categorical features (`ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`, `Sex`), dropping the first level of each to avoid multicollinearity.
4. **Split & Scale** — 80/20 stratified train/test split, then `StandardScaler` (fit on train only).
5. **Train & Compare** — fits all 5 models, ranks them by **F1 score** (a better metric than accuracy here, since a false negative on heart disease is costlier than a false positive).
6. **Report** — prints a full classification report for the top-ranked model.
7. **Persist** — 🎁 automatically saves the winning model, the fitted scaler, and the feature column order to `Classification/models/` using `joblib`:
   - `heart_best_model.pkl`
   - `heart_scaler.pkl`
   - `heart_feature_columns.pkl`

   This makes the model directly reusable for inference without re-training — just load the three `.pkl` files and transform new input the same way.

---

## 📊 Why Compare 5 Models?

Rather than picking one algorithm upfront, both notebooks train all five and let the data decide — a good habit for avoiding premature optimization. It also makes for a nice quick reference on how differently these algorithms behave:

| Model | Strengths | Watch out for |
|---|---|---|
| Logistic Regression | Fast, interpretable, great baseline | Assumes linear decision boundary |
| KNN | Simple, no training phase | Sensitive to feature scale & `k` choice |
| Naive Bayes | Very fast, works well with small data | Assumes feature independence |
| Decision Tree | Interpretable, handles non-linear splits | Prone to overfitting |
| SVM (RBF) | Strong on complex boundaries | Slower on larger datasets, less interpretable |

---

## ▶️ Running These Notebooks

From the repo root:

```bash
pip install -r requirements.txt
cd Classification
jupyter notebook
```

> ⚠️ `heart_disease_classification.ipynb` expects `heart.csv` to live at `../datasets/heart.csv` relative to this folder — keep the repo's folder structure intact.

---

## 🔗 Related Folders

- [`../EDA`](../EDA) — exploratory analysis of the Heart Disease dataset before this modeling step
- [`../Model_Evaluation`](../Model_Evaluation) — cross-validation & hyperparameter tuning that can be applied to sharpen these models further
