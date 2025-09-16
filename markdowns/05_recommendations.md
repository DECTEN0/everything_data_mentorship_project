# 04 recommendations

Goal: Predict which students are likely to graduate (class 1).

Compared models: Logistic Regression (baseline), Logistic Regression + SMOTE, and CatBoost.

Evaluation: Stratified K-fold CV, Accuracy, Precision, Recall, and F1-macro, with emphasis on class 1 recall/F1.

## Key Findings

Logistic Regression (no SMOTE): Low recall/F1 for graduates → struggled with class imbalance.

SMOTE + Logistic Regression: Improved F1-macro (~0.49) and recall for class 1 → better at identifying graduates, even with slightly lower accuracy than CatBoost.

CatBoost: Best overall accuracy (~0.61) but very poor recall for graduates (0.17) → unsuitable when missing graduates is costly.

## Model Recommendation

Use Logistic Regression with SMOTE as the primary model. It balances precision and recall and is significantly better at detecting students likely to graduate, which aligns with the project’s goal. 

CatBoost can be explored further, but only if improving class 1 recall (e.g., through class weights or threshold tuning).

## Improvements

1. Hyperparameter tuning: Try different SMOTE sampling strategies or Logistic Regression regularization settings.

2. Feature engineering: Add or refine features (e.g., attendance, grades) to improve signal for class 1(graduates).

3. Threshold optimization: Adjust decision thresholds to favor recall if identifying graduates is critical.

4. Alternative models: Test ensemble methods or tune CatBoost with class weights to improve minority class recall.

5. Data collection: Gather more examples of graduates to reduce imbalance.

For stakeholders:

Deploy the SMOTE + Logistic Regression model in a pilot setting to flag students at risk of not graduating. Pair predictions with targeted academic support, then monitor precision/recall metrics over time.
