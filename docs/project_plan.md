# Project Plan – Mentorship Cohort Analysis

## Phases & Deliverables
**Phase 0 – Setup**
- Create environment, repo, folder scaffold
- Add `cohort.csv` to `data/raw/`
- Define config in `configs/params.yaml`

**Phase 1 – Data Cleaning & Preprocessing**
- Validate schema and data types
- Handle missing values (median for numeric, mode for categorical)
- Standardize categorical labels (e.g., Gender, Age range)
- Engineer helpful features (e.g., hours-per-week buckets, regional grouping)

**Phase 2 – Exploratory Data Analysis**
- Distribution plots for demographics & motivations
- Correlations / associations to `Graduation status`
- Pivot tables by track, hours, score
- Save key visuals to `reports/figures/`

**Phase 3 – Predictive Modeling**
- Target: `Graduation status` (binary)
- Compare Logistic Regression vs Random Forest (5-fold CV on F1)
- Avoid leakage (consider excluding `Total score` at deployment time if unavailable prior to graduation decision)

**Phase 4 – Evaluation & Interpretation**
- Report accuracy, precision, recall, F1 on hold-out test set
- Feature importance / coefficients narrative
- Error analysis: false positives/negatives

**Phase 5 – Recommendations**
- Data-driven tactics to improve graduation rate (e.g., min weekly hours, targeted support)

**Phase 6 – Reporting & Handover**
- Executive summary in `reports/export/`
- Dashboards in `dashboards/` (Power BI / Tableau)
