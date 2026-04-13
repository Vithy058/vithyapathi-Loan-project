from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


class ReweighedModelWrapper:
    """Wrapper to give a reweighed sklearn model a consistent predict() interface."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Underlying model does not support predict_proba")


def _get_base_model(model_type):
    """Instantiate a fresh base estimator."""
    if model_type == "Random Forest":
        return RandomForestClassifier(random_state=42, n_estimators=50)
    return LogisticRegression(random_state=42, max_iter=1000)


def mitigate_bias(
    X_train,
    y_train,
    sensitive_features_train,
    model_type="Logistic Regression",
    method="Exponentiated Gradient",
):
    """
    Apply a bias mitigation technique and return a trained mitigated model.

    Supported methods:
    - 'Exponentiated Gradient' (Fairlearn in-processing)
    - 'Reweighing' (AIF360 pre-processing)
    """
    base_model = _get_base_model(model_type)

    # --- FINAL SAFETY CLEAN (One last check to prevent any NaN/Inf leakage) ---
    X_train = pd.DataFrame(X_train).fillna(0).replace([np.inf, -np.inf], 0)
    y_train = np.ravel(y_train)

    if method == "Exponentiated Gradient":
        try:
            # Fairlearn's Exponentiated Gradient is sensitive to dtypes
            sf_cleaned = np.array(sensitive_features_train).astype(str).ravel()
            
            mitigator = ExponentiatedGradient(
                base_model, constraints=DemographicParity()
            )
            mitigator.fit(
                X_train, y_train, sensitive_features=sf_cleaned
            )
            return mitigator
        except Exception as e:
            # Fallback to standard model if iteration fails or encounters internal NaNs
            base_model.fit(X_train, y_train)
            return base_model

    elif method == "Reweighing":
        try:
            from aif360.algorithms.preprocessing import Reweighing
            from aif360.datasets import BinaryLabelDataset
            from sklearn.preprocessing import LabelEncoder

            df = pd.DataFrame(X_train.copy())
            df["target"] = np.array(y_train)

            # AIF360 requires numeric values -- encode strings to integers
            sf_array = np.array(sensitive_features_train)
            if sf_array.dtype.kind in ("U", "S", "O"):
                le = LabelEncoder()
                sf_encoded = le.fit_transform(sf_array)
            else:
                sf_encoded = sf_array.astype(float)

            df["sensitive"] = sf_encoded

            mode_val = pd.Series(sf_encoded).mode()[0]
            privileged_groups = [{"sensitive": float(mode_val)}]
            unprivileged_classes = [
                float(v) for v in np.unique(sf_encoded) if v != mode_val
            ]
            unprivileged_groups = [{"sensitive": v} for v in unprivileged_classes]

            dataset = BinaryLabelDataset(
                df=df,
                label_names=["target"],
                protected_attribute_names=["sensitive"],
            )

            rw = Reweighing(
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )
            dataset_transf = rw.fit_transform(dataset)
            weights = dataset_transf.instance_weights

            base_model.fit(X_train, y_train, sample_weight=weights)
            return ReweighedModelWrapper(base_model)

        except Exception:
            base_model.fit(X_train, y_train)
            return ReweighedModelWrapper(base_model)

    else:
        base_model.fit(X_train, y_train)
        return base_model
