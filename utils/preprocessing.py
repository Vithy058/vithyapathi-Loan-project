import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import streamlit as st


def load_data(path):
    """Load a CSV dataset from disk."""
    return pd.read_csv(path)


@st.cache_data
def preprocess_data(df, target_col="loan_approval", sensitive_col=None):
    """
    Full preprocessing pipeline:
    - Separate features and target
    - Preserve raw sensitive features for fairness metrics
    - Impute missing values (median for numeric, mode for categorical)
    - Label-encode categorical columns
    - Standardize numeric columns
    """
    df = df.copy()

    # --- TARGET CONVERSION ---
    # Drop rows where target is missing COMPLETELY before doing anything
    df = df.dropna(subset=[target_col])
    
    y_raw = df[target_col]
    if isinstance(y_raw, pd.DataFrame):
        y_raw = y_raw.iloc[:, 0]
    
    # If target is numerical/continuous (like 'loan amount'), we convert to binary [0, 1]
    # to maintain compatibility with classification-based fairness metrics (Disparate Impact, etc.)
    target_was_continuous = False
    if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 10:
        median_val = y_raw.median()
        y = (y_raw > median_val).astype(int)
        target_was_continuous = True
    else:
        # Standard label encoding for discrete targets
        if not pd.api.types.is_numeric_dtype(y_raw):
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
        else:
            y = y_raw.values
    
    # Final safety check for 1D and Cleanliness
    y = np.ravel(y)

    X = df.drop(columns=[target_col]).copy()
    
    # Handle infinite values (convert to NaN so imputer can catch them)
    X = X.replace([np.inf, -np.inf], np.nan)

    sensitive_features_raw = None
    if sensitive_col and sensitive_col in df.columns:
        sf_raw = df[sensitive_col].copy()
        
        # --- IMPUTE SENSITIVE FEATURES ---
        # Fairlearn/Mitigation cannot handle NaNs in sensitive attributes
        if pd.api.types.is_numeric_dtype(sf_raw):
            imputer_sf = SimpleImputer(strategy="median")
        else:
            imputer_sf = SimpleImputer(strategy="most_frequent")
            
        # Reshape for imputer
        sf_reshaped = sf_raw.values.reshape(-1, 1)
        sf_imputed = imputer_sf.fit_transform(sf_reshaped)
        
        # Flatten back to original series shape
        sensitive_features_raw = pd.Series(sf_imputed.ravel(), index=sf_raw.index, name=sensitive_col)

    # Remove features that are purely IDs or unique strings (High Cardinality)
    # This prevents 'Columns must be same length' and memory issues
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            if X[col].nunique() > 100: # Threshold for categorical
                X = X.drop(columns=[col])

    # Drop columns that are entirely NaN
    X = X.dropna(axis=1, how='all')

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        imputed_data = num_imputer.fit_transform(X[num_cols])
        X[num_cols] = pd.DataFrame(imputed_data, columns=num_cols, index=X.index)

    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        imputed_cat = cat_imputer.fit_transform(X[cat_cols])
        X[cat_cols] = pd.DataFrame(imputed_cat, columns=cat_cols, index=X.index)

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Convert to string to avoid mixed types
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    sensitive_features = None
    if sensitive_col and sensitive_col in X.columns:
        sensitive_features = X[sensitive_col].copy()

    if len(num_cols) > 0:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, sensitive_features, encoders, sensitive_features_raw


@st.cache_data
def get_data_profile(df):
    """
    Generate a data profiling summary for the Data Management page.
    Returns a dict with row_count, col_count, missing_summary, dtypes_summary.
    """
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    profile = {
        "row_count": len(df),
        "col_count": len(df.columns),
        "missing_counts": missing.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "total_missing": int(missing.sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_cols": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_cols": df.select_dtypes(exclude=["number"]).columns.tolist(),
    }
    return profile
