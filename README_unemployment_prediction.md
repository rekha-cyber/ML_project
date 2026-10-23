# Predicting Next-Year Unemployment in European Democracies

**Ironhack | Data Science & Machine Learning — Project 2**  
**Task:** Supervised regression · **Tools:** Python, pandas, scikit-learn, Jupyter, Matplotlib/Seaborn, joblib

## Project overview

Can we predict a country's **unemployment rate in the following year** using its current economic conditions and the political composition of its government?

This project builds a reproducible machine-learning workflow using the **Comparative Political Data Set (CPDS)**. It compares regression models against simple baselines, evaluates the selected model on later years, and investigates whether political variables contribute useful predictive information.

**Main finding:** Ridge regression predicted next-year unemployment reasonably well, but did **not** outperform a simple *persistence baseline* on the held-out test period. Political features provided only a small additional signal in the final model.

## Research question and practical use

> How accurately can next year's unemployment rate be predicted from this year's unemployment, GDP growth, inflation, and government composition?

The intended use is **exploratory economic forecasting**: a researcher or policy analyst could compare model predictions with simple expectations. The model is **not** intended for operational policy decisions or causal claims about political parties.

The prediction target is a **continuous unemployment rate**, so this is a **regression** problem. The main metric is **mean absolute error (MAE)**, expressed in **percentage points of unemployment**. MAE is intuitive: an MAE of 1.0 corresponds to an average absolute difference of approximately one percentage point between the predicted and observed rates.

## Data

- **Source:** [Comparative Political Data Set (CPDS)](https://cpds-data.org/data/)
- **Coverage used:** 36 countries, with observations from 2000–2023 (subject to availability after target construction)
- **Unit of observation:** One country in one year
- **Target:** Unemployment rate in year **t + 1**
- **Predictors from year t:** Unemployment rate, real GDP growth, inflation, and the government-composition measures `gov_left`, `gov_centre`, and `gov_right`

Next-year unemployment is constructed **within each country**, so that the value for year *t* is matched to year *t + 1*, rather than to another country or a nonconsecutive year. The target is never used as an input feature.

**Data attribution:** CPDS, available from the project's official data page. Consult the source codebook for the dataset's definitions and recommended academic citation. **Reuse licence:** A specific licence has not been verified here; please check the source terms before redistributing the raw data.

## Methodology

### 1. Explore and prepare

The first notebook examines country-year coverage, variable distributions, missing values, and relationships between the predictors and the target. The data is arranged as a country-year panel, then the next-year target is created.

### 2. Split chronologically

Instead of randomly mixing observations from different years, the evaluation uses a **time-based split**:

| Split | Feature years | Purpose |
|---|---|---|
| Training | 2000–2018 | Cross-validation, model comparison, and tuning |
| Test | 2019–2022 | Final evaluation on later observations (predicting 2020–2023) |

This gives **684 training rows** and **142 test rows** in the prepared modelling data. A chronological split more closely reflects the forecasting task and reduces the risk of training on future years when predicting earlier years. It does not eliminate every potential source of dependence between country-year observations.

### 3. Establish baselines

Two simple reference approaches help determine whether machine learning adds value:

- **Dummy baseline:** Predict a constant learned from the training data.
- **Persistence baseline:** Predict that next year's unemployment will equal this year's unemployment: `unemployment_t+1 = unemployment_t`.

The persistence baseline is particularly important because unemployment rates tend to change gradually, so the current rate is already a useful predictor.

### 4. Build, compare, and tune models

The main preprocessing and modelling workflow uses scikit-learn pipelines, including **median imputation** for missing numeric values and **feature scaling** for models that need it. Model families were compared using cross-validation on the training data, and Ridge regression was selected and tuned.

| Approach | Cross-validation MAE (percentage points) |
|---|---:|
| Dummy baseline | 3.364 |
| Persistence baseline | 0.848 |
| Linear Regression | 0.758 |
| Ridge Regression (tuned) | **0.757** |
| Random Forest | 1.007 |

*Lower MAE is better. These are training-period cross-validation estimates, not test-set scores. Cross-validation results should be interpreted with the notebook's exact fold design in mind.*

Ridge only narrowly outperformed ordinary linear regression in cross-validation. Its regularization makes it a reasonable, comparatively simple final candidate, but the difference of **0.001 percentage points** should not be read as meaningful superiority.

## Final evaluation

The chosen Ridge pipeline was evaluated on the held-out later-year test data.

| Metric / model | Test result |
|---|---:|
| Ridge — MAE | **1.044 percentage points** |
| Ridge — RMSE | 1.339 percentage points |
| Ridge — R² | 0.741 |
| Persistence baseline — MAE | **0.768 percentage points** |
| Dummy baseline — MAE | 2.853 percentage points |

Ridge explained a substantial share of variation in the held-out targets (**R² = 0.741**) and beat the constant-value dummy baseline. **However, persistence achieved a lower test MAE than Ridge.** In this time period, simply carrying forward the current unemployment rate gave more accurate forecasts on average.

The gap between Ridge's cross-validation MAE (**0.757**) and test MAE (**1.044**) also illustrates why checking performance on later years matters.

## Do political features improve prediction?

To investigate whether government composition was materially influencing the Ridge model's predictions, the political features were shuffled in a diagnostic experiment while the other features remained unchanged.

| Experiment | MAE (percentage points) |
|---|---:|
| Original Ridge predictions | 1.044 |
| Political features shuffled | 1.070 |
| Change in MAE | **+0.026** |

Shuffling the political features made the predictions slightly worse. This suggests that the political variables contain **some predictive information in this fitted model**, but their contribution is small compared with the information available from economic conditions, particularly the current unemployment rate.

**Important:** This is a predictive sensitivity check, **not evidence that a government's political orientation causes changes in unemployment**. The political measures may also be correlated with one another or with economic and country-specific conditions.

## Interpretation and limitations

- **Strong persistence:** Current unemployment is a powerful starting point for forecasting unemployment one year ahead.
- **Performance varies across time:** A strong cross-validation result does not guarantee superior performance on later years.
- **Political signal is limited:** Disturbing political features produced only a small increase in Ridge's MAE.
- **Prediction is not causation:** The model cannot establish the effect of government ideology on economic outcomes.
- **Limited scope:** The results cover the available CPDS countries and years, not every country or future economic regime.
- **Dependence and shocks:** Country-year observations are not fully independent, and unexpected economic shocks can limit predictability.

**Conclusion:** The machine-learning workflow successfully produced and tested a reproducible regression model, but the **persistence baseline is the more accurate choice by test MAE for this dataset and evaluation period**. A worthwhile next step would be to model *changes* in unemployment or investigate country-specific patterns, always comparing against the persistence benchmark.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb
│   ├── 02_pipelines_and_models.ipynb
│   └── 03_evaluation_and_interpretation.ipynb
├── src/
│   └── functions.py
├── data/                  # place locally sourced data here
└── models/                # generated model artefacts
```

*The directory tree shows the intended project organization. Keep only folders and files that actually exist in the submitted repository.*

### Notebook guide

1. **`01_eda_and_baseline.ipynb`** — problem framing, data exploration, target construction, chronological split, and baseline analysis.
2. **`02_pipelines_and_models.ipynb`** — preprocessing, cross-validation, model comparison, Ridge tuning, and model selection.
3. **`03_evaluation_and_interpretation.ipynb`** — final test evaluation, error analysis, feature interpretation, political-feature shuffling, and model export.

## How to reproduce

1. Clone this repository and open a terminal in its root directory.
2. Create and activate a Python environment (recommended).
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Download the relevant CPDS data from the [official source](https://cpds-data.org/data/) and put it at the location expected by the first notebook. **Raw data is not included in this README.**
5. Launch Jupyter:

   ```bash
   jupyter lab
   ```

6. Run notebooks **01 → 02 → 03**, from top to bottom, adapting any local raw-data paths to your environment.

The trained pipeline is saved as **`models/ridge_unemployment.pkl`** in the project workflow. Reload it using `joblib` only in an environment you trust, and pass new data with the **same feature names, meanings, and units** used during training. Model artefacts generated by notebook 03 can be excluded from Git if the notebook recreates them.

## Presentation

The project presentation covers the prediction problem, features, baselines, model comparison, chronological test results, political-feature sensitivity analysis, and the limitations of drawing causal conclusions from prediction models.

## Data source

Comparative Political Data Set (CPDS): https://cpds-data.org/data/

For formal reuse, consult the CPDS website and codebook for version-specific attribution, methodology, and licensing terms.
