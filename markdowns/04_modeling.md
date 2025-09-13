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


```python
import os

os.chdir("..")
print("Current working dir:", os.getcwd())
#print("Files in raw folder:", os.listdir("data/raw"))
```

    Current working dir: C:\Users\Window\Desktop\Everything_Data_Mentorship\mentorship_ds_project
    

## c. Imports


```python
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
#import catboost as cb 
```

## d. Loading Data 


```python

preprocessor = joblib.load('artifacts/preprocessor.joblib')

# Load original splits
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()  # convert DataFrame to Series
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

# Transform again if needed
X_train_final = preprocessor.transform(X_train)
X_test_final = preprocessor.transform(X_test)
```

## e. Baseline Logistic Regression Model 


```python
#Logistic Regression

from sklearn.preprocessing import StandardScaler

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

Precision = 0.74, Recall = 1.00, F1 = 0.85 → The model is excellent at identifying class 0.

Precision = 0.00, Recall = 0.00, F1 = 0.00 → The model completely fails to detect class 1.

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

The model now predicts some graduates (class 1) as th recall improved from 0.00 → 0.17, which is progress for the minority class.

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
import numpy as np

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
    

We have moderate variability between folds – the standard deviation of ≈0.053 suggests performance is somewhat stable, but not highly reliable.

Low absolute F1-macro and accuracy – the classifier is struggling to separate graduates (class 1) from non-graduates (class 0).

Class imbalance likely affects performance – even with class_weight='balanced', the model is biased toward the majority class.

## g. Next Steps 

Trying a diffrent model; ie. Decision Tree–based models - CatBoost and Naïve Bayes - CategoricalNB model and compare the peformance to the logistic regression model.


```python

```
