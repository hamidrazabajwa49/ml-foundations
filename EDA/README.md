# 🔍 EDA (Exploratory Data Analysis)

> Digging into two datasets before any modeling happens — spotting bad data, visualizing distributions, checking correlations, and prepping clean, encoded features.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-Visualization-4c72b0)

---

## 📓 Notebooks

| Notebook | Dataset | Target | Rows | Features |
|---|---|---|---|---|
| [`heart_disease_eda.ipynb`](./heart_disease_eda.ipynb) | `../datasets/heart.csv` | `HeartDisease` (0/1) | 918 | 12 |
| [`insurance_eda.ipynb`](./insurance_eda.ipynb) | `../datasets/insurance.csv` | `charges` (continuous) | 1338 | 7 |

Both notebooks follow the same general arc:

```
Load → Inspect (shape, dtypes, nulls, duplicates) → Visualize → Clean → Encode → Scale
```

> ⚠️ `insurance_eda.ipynb` expects an `insurance.csv` file at `../datasets/insurance.csv`. Make sure it's added to the `datasets/` folder alongside `heart.csv`.

---

## ❤️ `heart_disease_eda.ipynb`

**Goal:** Understand the Heart Disease dataset and get it model-ready.

**What it covers:**
1. **Inspection** — shape, dtypes, summary stats, null check, duplicate check.
2. **Univariate analysis** — histograms with KDE for `Age`, `RestingBP`, `Cholesterol`, `MaxHR`.
3. **🚩 Data quality fix** — flags and repairs a real-world data issue: `Cholesterol` and `RestingBP` contain biologically impossible `0` values (a living person can't have zero cholesterol or blood pressure). These are recording errors, replaced with the column mean computed *excluding* the zeros — then the distributions are re-plotted to confirm the fix.
4. **Categorical vs. target** — count plots of `Sex`, `ChestPainType`, `FastingBS` split by `HeartDisease`, plus a boxplot of `Cholesterol` and a violin plot of `Age` by disease status.
5. **Correlation analysis** — full heatmap plus a ranked list of which features correlate most strongly with `HeartDisease`.
6. **Encoding & scaling** — one-hot encodes categorical columns and standard-scales the numeric ones (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`), producing a fully model-ready `df_encoded`.

**Key insight surfaced:** the zero-value cleanup step matters — without it, `Cholesterol` and `RestingBP` distributions are skewed by a spike of invalid zeros that would mislead any model trained on them.

---

## 💰 `insurance_eda.ipynb`

**Goal:** Explore what drives medical insurance charges and engineer useful features for later regression modeling.

**What it covers:**
1. **Inspection** — shape, dtypes, summary stats.
2. **Univariate analysis** — histograms for `age`, `bmi`, `children`, `charges`; boxplots to flag outliers; count plots for `children`, `sex`, `smoker`.
3. **Correlation heatmap** — quick look at linear relationships between numeric features and `charges`.
4. **Cleaning** — drops duplicate rows, checks for nulls.
5. **Encoding** — maps `sex` → `is_male` and `smoker` → `is_smoker` (0/1), one-hot encodes `region`.
6. **🧠 Feature engineering** — bins `bmi` into WHO-standard categories (`Underweight` <18.5, `Normal` 18.5–24.9, `Overweight` 25–29.9, `Obese` ≥30) and one-hot encodes the result.
7. **📐 Feature selection** — two complementary statistical tests:
   - **Pearson correlation** for numeric/binary features against `charges`
   - **Chi-Square test** (α = 0.05) for categorical features against binned `charges` quartiles, with an explicit **Keep/Drop** decision per feature
8. **Scaling** — standard-scales `age` and `bmi`, then reorders columns so `charges` sits last, producing a clean `df_cleaned` ready for regression.

**Key insight surfaced:** `is_smoker` is expected to dominate the Chi-Square results as the strongest predictor of charges, consistent with well-known findings on this dataset — the statistical test confirms it rather than assuming it.

---

## 🛠️ Techniques Used Across This Folder

| Category | Techniques |
|---|---|
| Inspection | `.info()`, `.describe()`, null/duplicate checks |
| Visualization | Histograms + KDE, boxplots, violin plots, count plots, correlation heatmaps |
| Data cleaning | Invalid-value imputation, deduplication |
| Encoding | Label mapping, one-hot encoding (`pd.get_dummies`) |
| Feature engineering | BMI binning (WHO categories) |
| Feature selection | Pearson correlation, Chi-Square test of independence |
| Scaling | `StandardScaler` |

---

## ▶️ Running These Notebooks

From the repo root:

```bash
pip install -r requirements.txt
cd EDA
jupyter notebook
```

---

## 🔗 Related Folders

- [`../datasets`](../datasets) — source data (`heart.csv`; add `insurance.csv` here too)
- [`../Classification`](../Classification) — heart disease EDA findings feed directly into `heart_disease_classification.ipynb`
- [`../Linear_Regression`](../Linear_Regression) — insurance EDA findings feed directly into `insurance_regression.ipynb`
