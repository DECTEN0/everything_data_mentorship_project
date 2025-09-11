import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# --- Configuration ---
DATA_PATH = 'data/your_dataset.csv' # Path to your CSV file
PREPROCESSOR_PATH = 'artifacts/preprocessor.joblib' # Path to saved preprocessor
MODEL_OUTPUT_PATH = 'artifacts/log_reg_baseline.joblib'
TARGET_COLUMN = 'graduated' # Replace with your target column name




def main():
    #Load the dataset
    df = pd.read_csv(DATA_PATH)
    #Separate features and target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    #Split the dataset (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
    )
    #Load the saved preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    #Transform the data
    X_train_final = preprocessor.transform(X_train)
    X_test_final = preprocessor.transform(X_test)
    #Train the Logistic Regression model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_final, y_train)
    #Evaluate the model
    y_pred = log_reg.predict(X_test_final)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    #Save the trained model
    joblib.dump(log_reg, MODEL_OUTPUT_PATH)
    print(f"Saved Logistic Regression model to {MODEL_OUTPUT_PATH}")




if __name__ == "__main__":
main()