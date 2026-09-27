# Machine Learning vs. Generalized Linear Models for Insurance Claim Frequency Prediction

An empirical comparison of a Poisson GLM, XGBoost, and LightGBM for motor insurance claim-frequency prediction, using the French Motor Third-Party Liability (freMTPL2freq) dataset.

> Bachelor Thesis — B.Sc. Data Science, IU International University of Applied Sciences
> Author: Tumisang Fokase

![Ordered Lorenz curves comparing risk-ranking performance of the GLM, XGBoost, and LightGBM](readme_assets/lorenz_curves.png)

## Overview

Insurance pricing depends on accurately predicting claim frequency. This project asks a simple question: **does the added flexibility of gradient-boosted trees produce a measurable advantage over a traditional Poisson GLM** — and if so, where does that advantage actually show up?

Three models are trained and evaluated on a common experimental setup:

- **Poisson GLM** — log-link, exposure offset, the traditional actuarial benchmark
- **XGBoost** — Poisson objective, exposure via `base_margin`
- **LightGBM** — Poisson objective, exposure via `init_score`

All three are compared across three complementary dimensions, not just one:

| Dimension | Metric(s) |
|---|---|
| Predictive accuracy | MAE, RMSE, Poisson deviance |
| Risk ranking | Gini coefficient (ordered Lorenz curve), bootstrap CIs |
| Claim/no-claim discrimination | ROC-AUC, precision, recall, F1 |

## Key Results

| Model | Poisson Deviance | Gini Coefficient | 95% CI |
|---|---|---|---|
| Poisson GLM | 0.314 | −0.002 | −0.015 to 0.010 |
| XGBoost | 0.301 | **0.103** | 0.085 to 0.113 |
| LightGBM | 0.301 | 0.101 | 0.082 to 0.108 |

**The headline finding:** all three models predict *average* claim frequency almost equally well. But when it comes to *ranking* policies from lowest to highest risk, the GLM performs no better than random (Gini ≈ 0), while XGBoost and LightGBM show a statistically significant advantage (bootstrap test, 1,000 resamples, 95% CI). SHAP analysis identifies `BonusMalus`, `VehAge`, and `DrivAge` as the most influential predictors for the gradient-boosting models — consistent with the largest-magnitude coefficients in the GLM.

**Takeaway:** the GLM remains a strong, interpretable baseline for average claim prediction. Gradient boosting earns its complexity specifically in risk stratification.

## Repository Structure

```
.
├── insurance_claim_prediction_comparative_analysis.ipynb   # Full analysis notebook
├── readme_assets/                                          # Images used in this README
└── README.md
```

## Dataset

[freMTPL2freq](https://www.openml.org/d/41214) — French motor third-party liability insurance data, obtained via OpenML (data ID 41214). Contains 670,000+ policy observations, including policyholder, vehicle, geographic, and exposure information.

## Methodology Summary

- **Preprocessing:** Exposure capped at 1 year, claims capped at 4. Population density log-transformed. `DrivAge`/`VehAge` are age-banded for the GLM but left continuous for the tree-based models.
- **Split:** 70/30 train-test, stratified on claim occurrence, fixed seed (42).
- **Evaluation:** Regression metrics (MAE, RMSE, Poisson deviance), risk-ranking (ordered Lorenz curve, Gini coefficient with bootstrap CIs), classification metrics with threshold selected via Youden's J statistic, and SHAP-based feature importance for XGBoost.

## Requirements

```bash
pip install pandas numpy scikit-learn statsmodels xgboost lightgbm shap matplotlib openml
```

## Running the Notebook

```bash
git clone <repo-url>
cd <repo-name>
jupyter notebook insurance_claim_prediction_comparative_analysis.ipynb
```

## Citation

If you use this work, please cite:

```
Fokase, T. (2026). Machine Learning vs. Generalized Linear Models for Insurance
Claim Frequency Prediction: An Empirical Comparison. Bachelor Thesis,
IU International University of Applied Sciences.
```

## License

Specify a license here (e.g., MIT) if you intend the code to be reused.
