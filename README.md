# End-to-End Credit Risk Scorecard & Production Lending Engine

A production-grade credit risk modeling and automated decisioning pipeline. This project covers the full end-to-end data science lifecycle: from exploratory data analysis and rigorous feature engineering to logistic regression modeling, FICO-style integer scorecard point allocation, P&L-driven financial threshold optimization, and real-time FastAPI deployment.

---

## Model Performance & Statistical Validation

* **Discriminatory Power:** **ROC-AUC of 0.8537** | **Gini Coefficient: 0.7074**
* **Kolmogorov-Smirnov (KS) Statistic:** **56.1%** (indicating exceptional separation between good and bad payers).
* **Risk Concentration (Decile Lift):** Validated via cumulative decile lift tables where lower risk deciles (D1–D3) contain minimal defaults, scaling up predictably to high-risk cohorts.
* **Financial Optimization:** P&L-driven threshold optimization identified an optimal default probability cutoff of **0.6491**, yielding an **88.82% approval rate** and **$24,164,700.00** in simulated net portfolio profit.

---

## Machine Learning Methodology & Architecture

### 1. Feature Engineering & Preprocessing
Raw tabular lending data underwent rigorous cleaning and transformation to handle heavy skews and non-linear relationships:
* **Log-Transformations:** Applied natural logarithms to highly skewed monetary and ratio features (`Log_Utilization`, `Log_DebtRatio`, `Log_MonthlyIncome`) to stabilize variance and normalize distributions for linear models.
* **Binary Flag Generation:** Created explicit risk indicators such as `IsHighUtilization` to capture non-linear default cliffs at high credit utilization thresholds.
* **Missing Value & Outlier Treatment:** Imputed missing values using robust statistical baselines and winsorized extreme outliers to prevent gradient distortion.

### 2. Predictive Modeling (Logistic Regression)
* **Algorithm Choice:** Implemented regularized Logistic Regression (`scikit-learn`), chosen for its interpretability, stability, and regulatory compliance in retail banking.
* **Class Imbalance Handling:** Addressed severe default-class imbalance using `class_weight='balanced'`, recalibrating the decision boundary and adjusting log-odds parameters to correctly penalize false negatives (defaulting borrowers).

### 3. FICO-Style Integer Scorecard Transformation
To translate abstract log-odds into an intuitive, transparent scoring system used by credit committees:
* Extracted raw model coefficients and intercept.
* Established a scaling baseline (e.g., Base Score **505.74** at a fixed odds ratio).
* Converted continuous logistic probabilities and feature weights into additive integer scorecard points, allowing risk managers to easily audit *why* an applicant received a specific score.

### 4. P&L-Driven Decision Optimization
Rather than relying on arbitrary statistical cutoffs (like 0.50), the model evaluates financial trade-offs:
* **Good Payer Revenue:** Captured net interest margin / profit per performing loan.
* **Default Loss:** Factorized Loss Given Default ($LGD$) on defaulted principals.
* **Threshold Sweep:** Scanned probability thresholds across a fine grid (`0.01` to `0.90`) to locate the global maximum profit point at **0.6491**.

