import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def detect_bias(y_test, y_pred, sensitive_features):
    if sensitive_features is None:
        return (None, None)
    
    # Standard Fairlearn metrics
    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)
    
    unique_groups = np.unique(sensitive_features)
    approval_rates = {}
    for group in unique_groups:
        mask = (sensitive_features == group)
        if np.sum(mask) > 0:
            approval_rates[str(group)] = float(np.mean(y_pred[mask]))
        else:
            approval_rates[str(group)] = 0.0
            
    if len(approval_rates) > 1:
        rates = list(approval_rates.values())
        min_rate = min(rates)
        max_rate = max(rates)
        disparate_impact = min_rate / max_rate if max_rate > 0 else 0.0
    else:
        disparate_impact = 1.0
        
    metrics = {
        'Demographic Parity Difference': float(dpd),
        'Equal Opportunity Difference': float(eod),
        'Disparate Impact': float(disparate_impact),
        'Statistical Parity Ratio': float(disparate_impact) # Often used interchangeably
    }
    return (metrics, approval_rates)

def detect_intersectional_bias(y_test, y_pred, df_sensitive):
    """
    Analyzes bias across combinations of sensitive attributes.
    df_sensitive: pd.DataFrame with multiple sensitive columns
    """
    if df_sensitive is None or df_sensitive.empty:
        return (None, None)
    
    # Robustly create intersectional groups (handles mixed types and NaNs)
    # Convert all columns to string, fill NaNs with "Unknown", and join
    intersectional_groups = df_sensitive.astype(str).replace('nan', 'Unknown').agg(' | '.join, axis=1)
    return detect_bias(y_test, y_pred, intersectional_groups)

def classify_risk(disparate_impact):
    if disparate_impact < 0.8:
        return ('High Risk', '#DC2626')
    elif disparate_impact < 0.9:
        return ('Moderate Risk', '#F59E0B')
    else:
        return ('Fair', '#16A34A')
