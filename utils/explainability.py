import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _unwrap_model(model):
    """Extract the underlying sklearn estimator from wrappers."""
    if hasattr(model, "predictors_"):
        return model.predictors_[0]
    if hasattr(model, "model"):
        return model.model
    return model


def compute_shap_values(model, X_train, X_test, model_type="Logistic Regression"):
    """
    Compute SHAP values for the given model and return (shap_values, explainer).
    Uses a subsample of X_test to keep computation fast.
    """
    shap_model = _unwrap_model(model)
    X_sample = X_test[:100] if len(X_test) > 100 else X_test

    # Configure explainer
    if model_type == "Random Forest":
        explainer = shap.TreeExplainer(shap_model)
    else:
        explainer = shap.LinearExplainer(shap_model, X_train)

    # --- COMPUTE SHAP ---
    shap_results = explainer.shap_values(X_sample)
    
    # --- STANDARDIZE OUTPUT DIMENSIONS ---
    # 1. New SHAP API returns an object with .values
    if hasattr(shap_results, "values"):
        shap_values = shap_results.values
    else:
        shap_values = shap_results
    
    # 2. Extract positive class for binary models (if output is 3D or List)
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_output = shap_values[1] # positive class
        else:
            shap_output = shap_values[0]
    elif len(shap_values.shape) == 3:
        # Array shape: (samples, features, classes)
        shap_output = shap_values[:, :, 1]
    else:
        shap_output = shap_values

    return shap_output, X_sample


def get_feature_importance(shap_values, feature_names):
    """
    Compute mean |SHAP| per feature and return sorted DataFrame.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_abs,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return importance_df


def generate_shap_summary_plot(model, X_train, X_test, model_type="Logistic Regression"):
    """Generate a matplotlib SHAP summary plot figure."""
    plt.close('all') # Clear previous figures
    
    # Reuse the robust extraction logic
    shap_vals, X_sample = compute_shap_values(model, X_train, X_test, model_type)
    
    # Capitalize columns for professional display
    X_sample_display = X_sample.copy()
    X_sample_display.columns = [c.replace('_', ' ').upper() for c in X_sample_display.columns]

    # Dynamically adjust height based on number of features
    num_features = X_sample.shape[1]
    fig_height = max(5, num_features * 0.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    try:
        # Generate the dots plot
        shap.summary_plot(
            shap_vals, 
            X_sample_display, 
            show=False,
            plot_type="dot",
            color_bar=True,
            plot_size=None # We control it with fig
        )
        plt.title("Feature Impact on Outcomes", fontsize=12, pad=20)
    except Exception as e:
        plt.text(0.5, 0.5, f"Plot generation issue: {e}", ha="center", va="center")

    plt.tight_layout(pad=3.0)
    return plt.gcf()
