from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import gradio as gr


#Load the Dataset

# ===== Defining the data directories ==== #
PROJECT_ROOT = Path(__file__).parent  # points to /home/user/app
DATA_DIR = PROJECT_ROOT / "data" / "processed"
data_path = DATA_DIR / "modeling_data.csv"

# ===== Define artifact and data file paths ==== #
df = pd.read_csv(data_path)

#Convert columns to categorical
def set_ordered_categories(df):
    """
    Converts specific columns of df to ordered categorical types for visualization.
    Returns a modified copy of the DataFrame.
    """

    # Define categories
    experience_categories = [
        "Less than six months",
        "6 months - 1 year",
        "1-3 years",
        "4-6 years"
    ]
    commitment_categories = ["less than 6 hours", "7-14 hours", "more than 14 hours"]
    skill_level_categories = ["Beginner", "Elementary", "Intermediate"]
    age_categories = ["18-24 years", "25-34 years", "35-44 years", "45-54 years"]

    # Map skill levels to shorter labels
    skill_map = {
        "Beginner - I have NO learning or work experience in data analysis/ data science": "Beginner",
        "Elementary - I have theoretical understanding of basic data analysis/ data science concepts": "Elementary",
        "Intermediate - I have theoretical knowledge and experience in data analysis/ data science": "Intermediate"
    }

    # Apply transformations
    df["years_experience"] = pd.Categorical(
        df["years_experience"], categories=experience_categories, ordered=True
    )
    df["weekly_commitment_hours"] = pd.Categorical(
        df["weekly_commitment_hours"], categories=commitment_categories, ordered=True
    )
    df["skill_level"] = pd.Categorical(
        df["skill_level"].replace(skill_map),
        categories=skill_level_categories,
        ordered=True
    )
    df["age_range"] = pd.Categorical(
        df["age_range"], categories=age_categories, ordered=True
    )

    return df

#Use function to convert to categorical
df = set_ordered_categories(df)

#Encode target feature "graduated to binary"
df['graduated'] = df['graduated'].map({'No': 0, 'Yes': 1})

#set the features and target variables
X = df.drop('graduated', axis=1)
y = df['graduated']

#Peform the train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Identify column groups
ordinal_cols = ['age_range', 'years_experience', 'weekly_commitment_hours', 'skill_level']
nominal_cols = ['gender', 'referral_source', 'track_applied', 'main_aim', 'aptitude_test_completed']
numerical_cols =['total_score']
#For OrdinalEncoder, I'll preserve the defined category order from the DataFrame
# I'll extract the categories directly from the categorical dtype
ordinal_categories = [X[col].cat.categories.tolist() for col in ordinal_cols]

# Build the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

#Instanciate SMOTE 
smote = SMOTE(random_state=42, k_neighbors=2)  # k_neighbors can be tuned

#Instatiate Logistic Regression
log_reg = LogisticRegression(
    max_iter=1000, 
    class_weight=None,
    C=1.0, #Best C
    random_state=42)
# Create a pipeline with preprocessing and model

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ('smote', smote),
        ('log_reg', log_reg)
    ]
)

# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Perform cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)


# Create the Gradio interface
def predict_graduation_status(age_range, gender, referral_source, years_experience,track_applied, weekly_commitment_hours, main_aim, skill_level,aptitude_test_completed, total_score):

    """
    Takes applicant details as input, converts them into a DataFrame,
    uses a trained model to predict the graduation status,
    and returns the prediction as a formatted string.
    """
  
    input_data = pd.DataFrame(
       {
           "age_range": [age_range], 
           "gender": [gender], 
           "referral_source": [referral_source], 
           "years_experience": [years_experience],
           "track_applied": [track_applied], 
           "weekly_commitment_hours": [weekly_commitment_hours], 
           "main_aim": [main_aim], 
           "skill_level": [skill_level],
           "aptitude_test_completed": [aptitude_test_completed], 
           "total_score": [total_score]
        }
    )
  
    prediction = model.predict(input_data)[0]
    return "Graduated" if prediction == 1 else "Did not graduate"


# Gradio Interface
iface = gr.Interface(
    fn=predict_graduation_status,
    inputs=[
        gr.Dropdown(
            ["18-24 years", "25-34 years", "35-44 years", "45-54 years"], 
            label="Age Range"
        ),
        gr.Dropdown(
            ["Male", "Female"], 
            label="Gender"
        ),
        gr.Dropdown(
            ['Word of mouth', 'WhatsApp', 'Twitter', 'LinkedIn', 'through a geeks for geeks webinar', 'Instagram', 'Friend'], 
            label="Referral Source"
        ),
        gr.Dropdown(
            ["Less than six months", "6 months - 1 year", "1-3 years", "4-6 years"],
            label="Years of Experience"
        ),
        gr.Dropdown(
            ["Data Science", "Data Analysis"], 
            label="Track Applied"
        ),
        gr.Dropdown(
            ["less than 6 hours", "7-14 hours", "more than 14 hours"],
            label="Weekly Commitment Hours"
        ),
        gr.Dropdown(
            ["Upskill", "Learn afresh", "Portfolio", "Networking", "Other"], 
            label="Main Aim"
        ),
        gr.Dropdown(
            ["Beginner", "Intermediate", "Elementary"], 
            label="Skill Level"
        ),
        gr.Dropdown(
            ["Yes", "No"], 
            label="Aptitude Test Completed"
        ),
        gr.Slider(
            minimum=0,
            maximum=100,
            step=1,
            label="Total Score"
        ),
    ],
    outputs=gr.Label(num_top_classes=2, label="Graduation Status"),
    title="Graduation Status Predictor",
    description="Provide applicant details to predict the likelihood of graduation."
)

iface.launch(share=True)
