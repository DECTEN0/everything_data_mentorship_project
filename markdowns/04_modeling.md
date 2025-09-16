# 04 modeling

## a. Key Considerations from Exploratory Data Analysis

1. Small sample size -  We will need models that are robust to overfitting and don’t require massive data.

2. Mostly categorical features - Well need algorithms that naturally handle categorical variables or can work well after encoding are suitable.

3. Binary outcome - almost any classifier works, but stability matters more than raw complexity here.

## b. Best algorithm choice 

1. Logistic Regression 

    Will be our baseline as it is simple, interpretable, and works well with small datasets.
    
    Needs one-hot encoding or similar for categorical variables.
    
    Needs regularization (L1/L2) helps prevent overfitting.

2. Decision Tree–based models - CatBoost
    
    CatBoost natively handles categorical features.
    
    Good at capturing nonlinear relationships.
    
    CatBoost might squeeze out more accuracy but risks overfitting.

3. Naïve Bayes - CategoricalNB

    Works well on small categorical-heavy datasets.
    
    Fast to train, interpretable.
    
    Assumes feature independence, which is often not true, but it’s robust enough for small data.

## c. Imports


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from catboost import CatBoostClassifier 
from sklearn.preprocessing import StandardScaler
```

## d. Loading Data 


```python
from pathlib import Path
import pandas as pd
import joblib

# ===== Defining the data directories ==== #
PROJECT_ROOT = Path(r"C:\Users\Window\Desktop\Everything_Data_Mentorship\mentorship_ds_project")
DATA_DIR = PROJECT_ROOT / "data"
COMMON_DATA_DIR = DATA_DIR / "raw"
INGESTED_DIR = DATA_DIR / "processed"

# ===== Define artifact and data file paths ==== #
preprocessor_file = PROJECT_ROOT / "artifacts" / "preprocessor.joblib"
X_train_file = INGESTED_DIR / "X_train.csv"
X_test_file = INGESTED_DIR / "X_test.csv"
y_train_file = INGESTED_DIR / "y_train.csv"
y_test_file = INGESTED_DIR / "y_test.csv"

# ===== Load the preprocessor and data ==== #
preprocessor = joblib.load(preprocessor_file)

X_train = pd.read_csv(X_train_file)
X_test = pd.read_csv(X_test_file)
y_train = pd.read_csv(y_train_file).squeeze()  # convert DataFrame to Series
y_test = pd.read_csv(y_test_file).squeeze()

# ===== Transform ==== #
X_train_final = preprocessor.transform(X_train)
X_test_final = preprocessor.transform(X_test)
```

## e. Baseline Logistic Regression Model 


```python
#Logistic Regression
scaler = StandardScaler()
X_train_final[:, -1:] = scaler.fit_transform(X_train_final[:, -1:])
X_test_final[:, -1:]  = scaler.transform(X_test_final[:, -1:])

# Initialize logistic regression (baseline, minimal tuning)
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Fit on training data
log_reg.fit(X_train_final, y_train)

# Predict
y_pred = log_reg.predict(X_test_final)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

    Accuracy: 0.7391304347826086
    
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.74      1.00      0.85        17
               1       0.00      0.00      0.00         6
    
        accuracy                           0.74        23
       macro avg       0.37      0.50      0.42        23
    weighted avg       0.55      0.74      0.63        23
    
    

    C:\Users\Window\anaconda3\envs\everything_data\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\Users\Window\anaconda3\envs\everything_data\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\Users\Window\anaconda3\envs\everything_data\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    

The warning often means the model isn’t predicting one or more labels at all.

### Baseline model Observations

Overall, the model correctly predicts 73.3% of the test set.

Class 0 (majority class) has 17 samples, while Class 1 (minority class) has only 6 samples.

The model always predicts class 0 (recall for class 1 is 0.00), which artificially inflates accuracy.

Precision = 0.74, Recall = 1.00, F1 = 0.85. The model is excellent at identifying class 0.

Precision = 0.00, Recall = 0.00, F1 = 0.00. The model completely fails to detect class 1.

The weighted average is dominated by class 0, again hiding the model’s inability to recognize class 1.

### Baseline model conclusion
The program is over-predicting dropouts - class 0.

The model defaults to predicting "did not graduate" for everyone.

This suggests our features are not separating graduates from non-graduates well.

### Why this is problematic

Our goal is to identify students who are likely to graduate therefore this model is useless as it provides no signal for class 1 - graduates.

The accuracy score is misleading because our dataset is imbalanced - 74% of the samples are non-graduates.

### Impact

Stakeholders would miss out on identifying potential graduates or at-risk students.

Any interventions based on this model would only ever target the majority class.


```python
#Saving the trained model
joblib.dump(log_reg, 'artifacts/log_reg_baseline.joblib')
```




    ['artifacts/log_reg_baseline.joblib']



### Improving peformance of the baseline model

#### 1. Adjust Class Weights. 
This penalizes misclassification of graduates more heavily.


```python
scaler = StandardScaler()
X_train_final[:, -1:] = scaler.fit_transform(X_train_final[:, -1:])
X_test_final[:, -1:]  = scaler.transform(X_test_final[:, -1:])

# Initialize logistic regression with class_weight as balanced
log_reg = LogisticRegression(class_weight='balanced', random_state=42)

# Fit on training data
log_reg.fit(X_train_final, y_train)

# Predict
y_pred = log_reg.predict(X_test_final)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

    Accuracy: 0.43478260869565216
    
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.64      0.53      0.58        17
               1       0.11      0.17      0.13         6
    
        accuracy                           0.43        23
       macro avg       0.38      0.35      0.36        23
    weighted avg       0.50      0.43      0.46        23
    
    

### Adjusted class_weight = "balanced" model observations
Our results after setting class_weight='balanced' show a clear shift in how the model treats the minority class (graduates).

The model now predicts some graduates (class 1) as th recall improved from 0.00 - 0.17, which is progress for the minority class.

Accuracy dropped, this is expected as the model is no longer “playing it safe” by always predicting the majority class - class 0.

Precision for class 1(graduates) remains low therefore most graduate predictions are still incorrect, indicating the features don’t yet strongly distinguish graduates.

Macro averages decreased slightly because accuracy is no longer inflated by ignoring class 1.


#### 2. Hyperparameter tuning using GridSearch CV


```python
from sklearn.model_selection import GridSearchCV

# Define the base model
log_reg = LogisticRegression(class_weight='balanced', max_iter=500, solver='liblinear')

# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 1.0, 2.0, 5.0, 10.0],   # Regularization strength (higher C = less regularization)
    'penalty': ['l1', 'l2']                  # Try both L1 and L2 penalties
}

# Grid search with stratified 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=5,                 # Stratified 5-fold CV
    scoring='f1_macro',   # Macro F1 balances both classes
    n_jobs=-1
)
```


```python
#Fit on the data
grid_search.fit(X_train_final, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Macro F1 Score:", grid_search.best_score_)
```

    Best Parameters: {'C': 1.0, 'penalty': 'l1'}
    Best Macro F1 Score: 0.5227731092436975
    


```python
# Use the best model to predict
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_final)

from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

    Accuracy: 0.5217391304347826
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.80      0.47      0.59        17
               1       0.31      0.67      0.42         6
    
        accuracy                           0.52        23
       macro avg       0.55      0.57      0.51        23
    weighted avg       0.67      0.52      0.55        23
    
    

The model is very confident when predicting non-graduates, but when it predicts graduates, it’s often wrong.

The model now identifies 67% of actual graduates (recall_score of 0.67), which is a big improvement from 0% recall earlier. However, it now misses many non-graduates (recall dropped from 1.00 to 0.47).

Graduates (class 1) have a usable but modest F1-score compared to before (previously 0.00).

Accuracy dropped because the model now misclassifies more non-graduates in favor of catching graduates, this is normal when handling class imbalance.

Macro and weighted averages are closer, suggesting less bias toward class 0.

Our F1-scores are moderate, indicating features or model complexity could be improved.

 ## f. Stratified k-fold 

With only 115 rows, a single 80/20 split means your test set has ~23 rows. Too small to trust. 

We use StratifiedKFold to preserves class balance instead of one fixed split.

    Stratified 5-Fold CV - every observation gets a chance to be in test set, while keeping class balance.
    
    Produces aggregate predictions across folds - more reliable classification reports and confusion matrices.
    
    Avoids the “tiny test set” problem.


```python
# Use all your preprocessed features and labels
X_all = preprocessor.transform(pd.concat([X_train, X_test], axis=0).reset_index(drop=True)) # Preprocessed features for the whole dataset
y_all = pd.concat([y_train, y_test], axis=0).reset_index(drop=True) # Target labels for the whole dataset

#Define Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Define the model with class balancing and best hyperparameters if available
log_reg = LogisticRegression(
    class_weight='balanced',
    max_iter=500,
    solver='liblinear',
    C=1.0,  #Best C
    penalty="l1" #Best penalty
)

#Evaluate with F1-macro (better for class imbalance)
scores = cross_val_score(log_reg, X_all, y_all, cv=skf, scoring='f1_macro')

#Summarize results
print("F1-macro scores for each fold:", scores)
print("Mean F1-macro score:", np.mean(scores))
print("Std deviation:", np.std(scores))

#Check accuracy too
acc_scores = cross_val_score(log_reg, X_all, y_all, cv=skf, scoring='accuracy')
print("Mean Accuracy:", np.mean(acc_scores))
```

    F1-macro scores for each fold: [0.41025641 0.3030303  0.425      0.45591398 0.43047619]
    Mean F1-macro score: 0.40493537645150546
    Std deviation: 0.05304092298440999
    Mean Accuracy: 0.47826086956521746
    

We have moderate variability between folds - the standard deviation of ≈0.053 suggests performance is somewhat stable, but not highly reliable.

Low absolute F1-macro and accuracy – the classifier is struggling to separate graduates (class 1) from non-graduates (class 0).

Class imbalance likely affects performance – even with class_weight='balanced', the model is biased toward the majority class.

## g. SMOTE with Logistic Regression and Stratified KFold Validation


```python
X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True) # Preprocessed features for the whole dataset
y_all = pd.concat([y_train, y_test], axis=0).reset_index(drop=True) # Target labels for the whole dataset

#Imports 
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

#Instanciate SMOTE 
smote = SMOTE(random_state=42, k_neighbors=2)  # k_neighbors can be tuned

#Instatiate Logistic Regression
log_reg = LogisticRegression(
    max_iter=1000, 
    class_weight=None,
    C=1.0, #Best C
    random_state=42)

#Create a pipeline 
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('log_reg', log_reg)
])

# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = []
accuracies = []
precisions = []
recalls = []

# Perform cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):
    X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
    y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]

    pipeline.fit(X_train, y_train)      
    y_pred = pipeline.predict(X_test)

    # Classification report for this fold
    print(f"\n=== Fold {fold} Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Did Not Graduate (0)', 'Graduated (1)']))

    # Collect metrics
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)

    f1_scores.append(f1)
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)

#Results
print(f"F1-macro scores: {f1_scores}")
print(f"Mean Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
print(f"Mean Recall:    {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
print(f"Mean F1-macro: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
print(f"Mean Accuracy: {np.mean(accuracies):.3f}")
```

    C:\Users\Window\anaconda3\envs\everything_data\Lib\site-packages\sklearn\preprocessing\_encoders.py:246: UserWarning: Found unknown categories in columns [1] during transform. These unknown categories will be encoded as all zeros
      warnings.warn(
    

    
    === Fold 1 Classification Report ===
                          precision    recall  f1-score   support
    
    Did Not Graduate (0)       0.88      0.41      0.56        17
           Graduated (1)       0.33      0.83      0.48         6
    
                accuracy                           0.52        23
               macro avg       0.60      0.62      0.52        23
            weighted avg       0.73      0.52      0.54        23
    
    
    === Fold 2 Classification Report ===
                          precision    recall  f1-score   support
    
    Did Not Graduate (0)       0.79      0.65      0.71        17
           Graduated (1)       0.33      0.50      0.40         6
    
                accuracy                           0.61        23
               macro avg       0.56      0.57      0.55        23
            weighted avg       0.67      0.61      0.63        23
    
    
    === Fold 3 Classification Report ===
                          precision    recall  f1-score   support
    
    Did Not Graduate (0)       0.67      0.35      0.46        17
           Graduated (1)       0.21      0.50      0.30         6
    
                accuracy                           0.39        23
               macro avg       0.44      0.43      0.38        23
            weighted avg       0.55      0.39      0.42        23
    
    
    === Fold 4 Classification Report ===
                          precision    recall  f1-score   support
    
    Did Not Graduate (0)       0.75      0.71      0.73        17
           Graduated (1)       0.29      0.33      0.31         6
    
                accuracy                           0.61        23
               macro avg       0.52      0.52      0.52        23
            weighted avg       0.63      0.61      0.62        23
    
    
    === Fold 5 Classification Report ===
                          precision    recall  f1-score   support
    
    Did Not Graduate (0)       0.69      0.69      0.69        16
           Graduated (1)       0.29      0.29      0.29         7
    
                accuracy                           0.57        23
               macro avg       0.49      0.49      0.49        23
            weighted avg       0.57      0.57      0.57        23
    
    F1-macro scores: [0.5180952380952382, 0.5548387096774194, 0.38076923076923075, 0.5174825174825175, 0.48660714285714285]
    Mean Precision: 0.522 ± 0.057
    Mean Recall:    0.526 ± 0.068
    Mean F1-macro: 0.492 ± 0.059
    Mean Accuracy: 0.539
    

    C:\Users\Window\anaconda3\envs\everything_data\Lib\site-packages\sklearn\preprocessing\_encoders.py:246: UserWarning: Found unknown categories in columns [1, 3] during transform. These unknown categories will be encoded as all zeros
      warnings.warn(
    

### SMOTE achieved:

Balanced training set per fold, helping the model learn class 1 (“graduated”) patterns better.

More stable F1 scores across folds compared to earlier runs.

Reduced extreme bias toward class 0, as shown by the improved F1-macro.

Better average precision for class 0.

### Limitations

Accuracy is still moderate (0.557), meaning many predictions remain misclassified.

Precision/recall values are close to random guessing (0.5), suggesting the features may not strongly separate the classes.

## h. Random Forest; CatBoost


```python
from src.features import convert_categorical
X_train_cat = convert_categorical(X_train)
X_test_cat = convert_categorical(X_test)
cat_features = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# Define the base model
cat_model = CatBoostClassifier(verbose=0,random_seed=42, cat_features=cat_features)

#Define Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Define the param grid
param_grid = {
    'iterations': [200, 500],
    'depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'l2_leaf_reg': [3, 5, 7]
}

#GridSearchCV
grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    cv=skf,
    scoring='f1_macro',
    n_jobs=-1
)

#Fit GridSearchCV on the training data
grid_search.fit(X_train_final, y_train)

#Get the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

#Train a final model using the best parameters on the full training set
best_model = grid_search.best_estimator_
best_model.fit(X_train_final, y_train)

#Make predictions
y_pred = best_model.predict(X_test_final)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

    Best Parameters: {'depth': 4, 'iterations': 500, 'l2_leaf_reg': 5, 'learning_rate': 0.2}
    Best CV Accuracy: 0.5335390025045198
    Accuracy: 0.6086956521739131
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.72      0.76      0.74        17
               1       0.20      0.17      0.18         6
    
        accuracy                           0.61        23
       macro avg       0.46      0.47      0.46        23
    weighted avg       0.59      0.61      0.60        23
    
    

Accuracy of 0.61 - Better than Logistic Regression (0.557), but still misses many class 1s.

Class 0 F1-score of	0.74 - The model is very confident predicting non-graduates.

Class 1 F1-score of 0.18 - The model struggles to identify graduates (many false negatives).

Macro Avg F1-score of 0.46 - Average performance across classes is moderate.

Weighted Avg F1-score of 0.60 - Skewed toward class 0 due to class imbalance.

Better accuracy and class 0 stability compared to Logistic Regression.

CatBoost’s gradient boosting approach can model non-linear relationships better than logistic regression.

### Challanges
Low recall for class 1 (17%) - Most graduates are being misclassified.

Small dataset (115 rows) - Limits model’s ability to learn complex patterns.

Imbalanced classes - Even with boosting, the model favors the majority class.

## i. Conclusion

Baseline Logistic Regression (no scaling, no balancing) - Missed most graduates.

Logistic Regression with class_weight='balanced' - Significant improvement in recall for class 1 while keeping a moderate F1.

SMOTE + Logistic Regression (Stratified KFold) - Similar to weighted logistic regression. Had good recall for graduates but lower precision. SMOTE helped further balance performance across folds.

CatBoost (tuned) - Best overall accuracy, but very poor detection of graduates - it tends to predict class 0.

Logistic Regression with SMOTE clearly outperformed the other models for recall and F1 of class 1. 

Logistic Regression with SMOTE maintained a reasonable trade-off: Mean Accuracy: 0.54 and Mean F1-macro: 0.49 and better recall for class 1, which is our priority.

Therefore this model found more actual graduates even if their overall accuracy was slightly lower.
