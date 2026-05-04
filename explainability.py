"""
LoanGuard AI — File A: explainability.py
=========================================
AI Explainability & Proxy Detection Module

This module is responsible for analyzing the LoanGuard AI model to provide:
1. Global feature importance (using SHAP TreeExplainer).
2. Individual prediction explanations.
3. Proxy variable detection via feature correlation.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"  # Clean format for visibility
)
logger = logging.getLogger(__name__)


class LoanExplainabilityAnalyzer:
    """
    A class to perform explainability analysis and proxy variable detection
    on a trained machine learning model for loan approval.
    """

    FEATURE_NAMES = [
        "age", "gender", "income", "credit_score", "loan_amount"
    ]

    def __init__(
        self,
        model_path: Union[str, Path] = "artifacts/loan_model.pkl",
        x_test_path: Union[str, Path] = "artifacts/X_test.csv",
        y_test_path: Union[str, Path] = "artifacts/y_test.csv",
        output_dir: Union[str, Path] = "outputs/explainability"
    ):
        """
        Initializes the analyzer with paths to model and datasets.

        Args:
            model_path: Path to the trained pickled model.
            x_test_path: Path to the test features CSV.
            y_test_path: Path to the test labels CSV.
            output_dir: Directory where generated plots will be saved.
        """
        self.model_path = Path(model_path)
        self.x_test_path = Path(x_test_path)
        self.y_test_path = Path(y_test_path)
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model: Any = None
        self.x_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.shap_values: Optional[np.ndarray] = None

    def _generate_artifacts(self):
        logger.info("Artifacts not found. Generating mock artifacts from data/loan_data.csv...")
        import pandas as pd
        import pickle
        from utils.preprocessing import preprocess_data
        from utils.training import train_model
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv("data/loan_data.csv")
        X, y, _, _, sf_raw = preprocess_data(df, target_col="loan_approval", sensitive_col="gender")
        model, metrics, X_test, y_test, y_pred, X_train, y_train, sf_train, sf_test = train_model(X, y, sensitive_features=sf_raw, model_type="Random Forest")
        
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)
        
        X_test.to_csv(self.x_test_path, index=False)
        pd.Series(y_test, name="loan_approval").to_csv(self.y_test_path, index=False)
        logger.info("Mock artifacts generated successfully.")

    def load_artifacts(self) -> None:
        """Loads the saved model and test datasets into memory."""
        logger.info("Loading artifacts...")
        
        if not self.model_path.exists() or not self.x_test_path.exists() or not self.y_test_path.exists():
            self._generate_artifacts()

        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            
            self.x_test = pd.read_csv(self.x_test_path)
            self.y_test = pd.read_csv(self.y_test_path).squeeze()
            
            logger.info(f"Test samples loaded: {len(self.x_test)}")
            logger.info(f"Test set approval rate: {self.y_test.mean():.1%}")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise

    def compute_global_importance(self) -> None:
        """Computes global SHAP feature importance and saves the summary plot."""
        logger.info("\n[1/3] Computing SHAP values (this may take a moment)...")
        if self.model is None or self.x_test is None:
            raise ValueError("Model and test data must be loaded first.")

        explainer = shap.TreeExplainer(self.model)
        shap_vals = explainer.shap_values(self.x_test)

        # Handle binary classification lists vs arrays vs 3D arrays
        if isinstance(shap_vals, list):
            if len(shap_vals) > 1:
                self.shap_values = shap_vals[1]
            else:
                self.shap_values = shap_vals[0]
        elif len(shap_vals.shape) == 3:
            self.shap_values = shap_vals[:, :, 1]
        else:
            self.shap_values = shap_vals

        # Mean absolute SHAP per feature
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": self.FEATURE_NAMES,
            "Mean |SHAP|": mean_shap,
        }).sort_values("Mean |SHAP|", ascending=False)

        logger.info("\n--- Global Feature Importance (SHAP) ---\n%s", shap_df.to_string(index=False))

        # Plot saving
        self._save_global_importance_plot(shap_df)

    def _save_global_importance_plot(self, shap_df: pd.DataFrame) -> None:
        """Generates and saves the horizontal bar chart for global SHAP values."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(shap_df["Feature"][::-1], shap_df["Mean |SHAP|"][::-1], color="steelblue")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("LoanGuard AI — Global Feature Importance (SHAP)")
        plt.tight_layout()
        
        out_path = self.output_dir / "shap_global_importance.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info(f"\nSaved global importance plot -> {out_path}")

    def explain_individual_prediction(self, sample_idx: int = 0) -> None:
        """Explains an individual prediction using SHAP values."""
        logger.info(f"\n[2/3] Generating individual explanation for sample #{sample_idx}...")
        
        if self.model is None or self.x_test is None or self.shap_values is None:
            raise ValueError("Model, test data, and SHAP values must be loaded and computed first.")

        sample = self.x_test.iloc[[sample_idx]]
        pred_prob = self.model.predict_proba(sample)[0][1]
        pred_label = self.model.predict(sample)[0]
        sample_sv = self.shap_values[sample_idx]

        logger.info(f"Prediction: {'APPROVED' if pred_label == 1 else 'REJECTED'}")
        logger.info(f"Approval Probability: {pred_prob:.2%}\n")

        contrib_df = pd.DataFrame({
            "Feature": self.FEATURE_NAMES,
            "Value": sample.values[0],
            "SHAP": sample_sv,
        }).sort_values("SHAP", ascending=False)

        self._log_individual_contributions(contrib_df)
        self._save_individual_waterfall_plot(contrib_df, sample_idx, pred_label, pred_prob)

    def _log_individual_contributions(self, contrib_df: pd.DataFrame) -> None:
        """Helper to log individual SHAP contributions clearly."""
        for _, row in contrib_df.iterrows():
            direction = "▲ pushes toward APPROVE" if row["SHAP"] > 0 else "▼ pushes toward REJECT"
            logger.info(f"  {row['Feature']:<20} = {row['Value']:>10.2f} | SHAP = {row['SHAP']:+.4f} {direction}")

    def _save_individual_waterfall_plot(
        self, contrib_df: pd.DataFrame, sample_idx: int, pred_label: int, pred_prob: float
    ) -> None:
        """Generates and saves the individual feature contribution bar chart."""
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["green" if v > 0 else "red" for v in contrib_df["SHAP"]]
        ax.barh(contrib_df["Feature"], contrib_df["SHAP"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on prediction)")
        
        status = "APPROVED" if pred_label else "REJECTED"
        ax.set_title(f"Individual Explanation — Sample #{sample_idx}\nPrediction: {status} ({pred_prob:.1%})")
        plt.tight_layout()
        
        out_path = self.output_dir / f"shap_individual_sample{sample_idx}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info(f"\nSaved individual explanation plot -> {out_path}")

    def analyze_proxy_variables(self) -> None:
        """Calculates feature correlations to detect potential proxy variables."""
        logger.info("\n[3/3] Running proxy variable detection (Correlation Matrix)...")
        if self.x_test is None:
            raise ValueError("Test data must be loaded first.")

        corr = self.x_test.corr(method="pearson")
        logger.info("\n--- Pearson Correlation Matrix ---\n%s", corr.round(3).to_string())

        # Check proxy risk against sensitive group (e.g., gender)
        if "gender" in corr.columns:
            gender_corr = corr["gender"].drop("gender").abs().sort_values(ascending=False)
            logger.info("\nFeatures correlated with 'gender' (Proxy Risk Check):")
            for feat, val in gender_corr.items():
                flag = " ⚠ PROXY RISK" if val > 0.15 else ""
                logger.info(f"  {feat:<20} |r| = {val:.4f}{flag}")

        # Save heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax
        )
        ax.set_title("LoanGuard AI — Feature Correlation Matrix\n(Proxy Detection)")
        plt.tight_layout()
        
        out_path = self.output_dir / "correlation_matrix.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info(f"\nSaved correlation matrix heatmap -> {out_path}")

    def run_full_analysis(self) -> None:
        """Executes the entire end-to-end explainability pipeline."""
        logger.info("=" * 60)
        logger.info("  LoanGuard AI — Explainability Report")
        logger.info("=" * 60)
        
        try:
            self.load_artifacts()
            self.compute_global_importance()
            self.explain_individual_prediction(sample_idx=0)
            self.analyze_proxy_variables()
            
            logger.info("\n" + "=" * 60)
            logger.info("Explainability analysis completed successfully.")
            logger.info(f"All reports and charts saved to: {self.output_dir}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")


if __name__ == "__main__":
    analyzer = LoanExplainabilityAnalyzer()
    analyzer.run_full_analysis()
