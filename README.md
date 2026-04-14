# 🛡️ LoanGuard: AI Fairness Audit & Bias Mitigation Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**LoanGuard** is an enterprise-grade compliance platform designed to audit, detect, and mitigate algorithmic bias in loan approval models. It provides a comprehensive suite of tools for financial institutions to ensure regulatory compliance (ECOA, GDPR, EU AI Act) and promote ethical AI lending practices.

---

## 🌟 Key Features

- **📊 Advanced Data Profiling**: Seamlessly ingestion and profiling of loan datasets with automated demographic attribute detection.
- **🤖 Robust Model Training**: Integrated training pipeline with support for various classification models.
- **🔍 Bias Detection Engine**: Deep-dive analysis using metrics like Disparate Impact, Demographic Parity, and Equalized Odds.
- **🛠️ Automated Mitigation**: Implementation of fairness-aware retraining algorithms to equalize approval rates across protected groups.
- **📈 Performance Comparison**: Before-versus-after analysis to visualize the precision-fairness trade-off.
- **💡 SHAP Explainability**: Global and local feature importance using SHAP values to explain every decision.
- **📄 Regulatory Reporting**: One-click generation of PDF compliance reports for auditors and regulators.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Prakash-Ramakrishnan110/Loan-project.git
   cd Loan-project
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (High-performance web dashboard)
- **Data Engine**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualizations**: [Plotly](https://plotly.com/), [Graph Objects](https://plotly.com/python/graph-objects/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
- **Explainability**: [SHAP](https://shap.readthedocs.io/)
- **Compliance**: [KaggleHub](https://github.com/Kaggle/kagglehub) for real-time dataset synchronization.

---

## 📋 Regulatory Framework Compliance

The platform is designed to assist in meeting the requirements of:
- **ECOA**: Equal Credit Opportunity Act
- **FHA**: Fair Housing Act
- **GDPR**: General Data Protection Regulation (Right to Explanation)
- **EU AI Act**: High-Risk System Requirements

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created with ❤️ by [Prakash Ramakrishnan](https://github.com/Prakash-Ramakrishnan110)*
