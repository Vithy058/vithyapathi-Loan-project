import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


def detect_bias(y_test, y_pred, sensitive_features):
    """
    Compute fairness metrics given ground-truth labels, predictions,
    and the sensitive attribute vector.

    Returns
    -------
    metrics : dict with DPD, EOD, DI
    approval_rates : dict mapping group -> approval rate
    """
    if sensitive_features is None:
        return None, None

    dpd = demographic_parity_difference(
        y_test, y_pred, sensitive_features=sensitive_features
    )
    eod = equalized_odds_difference(
        y_test, y_pred, sensitive_features=sensitive_features
    )

    unique_groups = np.unique(sensitive_features)
    approval_rates = {}

    for group in unique_groups:
        mask = sensitive_features == group
        if np.sum(mask) > 0:
            approval_rates[str(group)] = float(np.mean(y_pred[mask]))
        else:
            approval_rates[str(group)] = 0.0

    if len(approval_rates) > 0:
        min_rate = min(approval_rates.values())
        max_rate = max(approval_rates.values())
        disparate_impact = min_rate / max_rate if max_rate > 0 else 0.0
    else:
        disparate_impact = 1.0

    metrics = {
        "Demographic Parity Difference": float(dpd),
        "Equal Opportunity Difference": float(eod),
        "Disparate Impact": float(disparate_impact),
    }

    return metrics, approval_rates


def classify_risk(disparate_impact):
    """
    Classify fairness risk level based on disparate impact ratio.

    Returns (label, color_hex)
    """
    if disparate_impact < 0.8:
        return "High Risk", "#DC2626"
    elif disparate_impact < 0.9:
        return "Moderate Risk", "#F59E0B"
    else:
        return "Fair", "#16A34A"
