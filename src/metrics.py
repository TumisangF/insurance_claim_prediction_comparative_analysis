"""
Evaluation metrics for comparing claim-frequency prediction models.

Provides a canonical Gini / Lorenz-curve implementation and a shared set of
regression and classification metrics, used identically across the GLM,
XGBoost, and LightGBM models in this thesis so that results are directly
comparable.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)


def gini_and_lorenz(y_true, pred_counts, exposure_vals):
    """
    Ordered Lorenz curve and normalized Gini coefficient for risk ranking.

    Policies are sorted by predicted claim count in ascending order, and the
    cumulative share of exposure (x-axis) is plotted against the cumulative
    share of actual claims (y-axis). A model with genuine risk-ranking
    ability produces a curve below the diagonal (area under the curve < 0.5),
    so the Gini coefficient is defined as:

        Gini = 1 - 2 * (area under the Lorenz curve)

    This convention gives a positive value for a model with real
    discriminatory power and a value close to zero for a model equivalent to
    random ranking.

    Parameters
    ----------
    y_true : array-like
        Actual claim counts.
    pred_counts : array-like
        Predicted claim counts (predicted frequency x exposure).
    exposure_vals : array-like
        Policy exposure, in years.

    Returns
    -------
    gini : float
    cum_claims : np.ndarray
    cum_exposure : np.ndarray
    """
    y_true = np.asarray(y_true)
    pred_counts = np.asarray(pred_counts)
    exposure_vals = np.asarray(exposure_vals)

    order = np.argsort(pred_counts)
    sorted_actual = y_true[order]
    sorted_exposure = exposure_vals[order]

    cum_claims = np.cumsum(sorted_actual) / np.sum(sorted_actual)
    cum_exposure = np.cumsum(sorted_exposure) / np.sum(sorted_exposure)

    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    area = trapezoid(cum_claims, cum_exposure)
    gini = 1 - 2 * area
    return gini, cum_claims, cum_exposure


def evaluate_predictions(y_true, pred_counts, exposure_vals, model_name):
    """
    Regression metrics (MAE, RMSE, Poisson deviance, Gini coefficient) for a
    single model.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_counts = np.asarray(pred_counts, dtype=float)
    exposure_vals = np.asarray(exposure_vals, dtype=float)

    mae = mean_absolute_error(y_true, pred_counts)
    rmse = np.sqrt(mean_squared_error(y_true, pred_counts))

    eps = 1e-10
    poisson_dev = 2 * np.mean(
        y_true * np.log((y_true + eps) / (pred_counts + eps)) - (y_true - pred_counts)
    )

    gini, _, _ = gini_and_lorenz(y_true, pred_counts, exposure_vals)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "Poisson Deviance": poisson_dev,
        "Gini Coefficient": gini,
        "Mean Predicted": pred_counts.mean(),
        "Mean Actual": y_true.mean(),
    }


def claim_probability(pred_counts):
    """
    Convert predicted claim counts to a predicted probability of at least
    one claim, under a Poisson assumption: P(N >= 1) = 1 - exp(-mu).
    """
    pred_counts = np.asarray(pred_counts, dtype=float)
    return 1 - np.exp(-pred_counts)


def youden_optimal_threshold(y_true_binary, y_prob):
    """
    Select a decision threshold via Youden's J statistic
    (J = sensitivity + specificity - 1), i.e. the point on the ROC curve
    that jointly maximizes the true-positive and true-negative rates.

    A fixed threshold of 0.5 is not meaningful for this dataset: the
    predicted probability of at least one claim rarely exceeds ~0.2 given
    the overall claim rate, so a 0.5 cutoff would classify every policy as
    "no claim". Youden's J is a standard, threshold-free-selection approach
    for imbalanced binary classification problems.
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def classification_metrics(y_true, pred_counts, model_name, threshold=None):
    """
    Binary classification metrics (claim vs. no claim) derived from count
    predictions via a Poisson probability-of-at-least-one-claim transform.

    If `threshold` is not supplied, it is selected per model via Youden's J
    statistic on the ROC curve (see `youden_optimal_threshold`).
    """
    y_true_binary = (np.asarray(y_true) > 0).astype(int)
    y_prob = claim_probability(pred_counts)

    if threshold is None:
        threshold = youden_optimal_threshold(y_true_binary, y_prob)

    y_pred_binary = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true_binary, y_pred_binary)

    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true_binary, y_pred_binary),
        "Precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
        "Recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
        "F1 Score": f1_score(y_true_binary, y_pred_binary, zero_division=0),
        "ROC AUC": roc_auc_score(y_true_binary, y_prob),
        "Confusion Matrix": cm,
    }
