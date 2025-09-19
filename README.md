# Everything Data Mentorship Project – Graduation Prediction  

## Project Overview  
This project analyzes data from a previous **Everything Data Mentorship** cohort to understand participant demographics, motivations, and performance. The main goal is to **predict graduation status** using key features like experience, weekly commitment hours, and total score, then provide actionable recommendations to improve graduation rates.  

---

## Dataset Overview  
The dataset contains mentorship participant information, including:  

| Column | Description |
|--------|-------------|
| Timestamp | Submission time of participant response |
| ID No. | Unique participant identifier |
| Age range | Age group of the participant |
| Gender | Gender of the participant |
| Country | Participant’s country |
| Referral source | How the participant heard about Everything Data |
| Years of learning experience | Experience in data-related fields |
| Track applied for | Chosen data track |
| Weekly commitment hours | Hours per week available for the program |
| Main aim for joining | Participant’s primary goal |
| Motivation for joining | Reason for joining |
| Skill level | Self-assessed skill level |
| Aptitude test completion status | Whether the aptitude test was completed |
| Total score | Aptitude test score (58.33–86) |
| Graduation status | **Target variable** (1 = Graduated, 0 = Did not graduate) |

---

## Objectives  
- Perform **data cleaning and preprocessing** using Python (Pandas, NumPy).  
- Conduct **exploratory data analysis** (Matplotlib, Seaborn) to visualize distributions and correlations.  
- Build **classification models** (e.g., Logistic Regression, CatBoost) to predict graduation status.  
- Evaluate models using **accuracy, precision, recall, and F1-score**.  
- Recommend data-driven strategies to **improve graduation rates**.  

---

## Tools & Libraries  
- **Python 3.10+**  
- **Pandas**, **NumPy** – Data cleaning and manipulation  
- **Matplotlib**, **Seaborn** – Exploratory data analysis and visualization  
- **Scikit-learn** – Preprocessing, Logistic Regression, model evaluation  
- **CatBoost** – Advanced gradient boosting classifier  
- **Imbalanced-learn (SMOTE)** – Address class imbalance  
- **Jupyter Notebook** – Interactive development environment
- **Gradio** – Web-based model interface for deployment  

---

## Workflow  
1. **Data Overview**  
   - Basic Data Exploration.
   - Renamed column names to short descriptive names.  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized participant demographics and test score distributions.  
   - Investigated relationships between features (e.g., skill level vs. graduation).  

3. **Feature Engineering**  
   - Handled missing values using mode and median where appropriate.  
   - Encoded categorical variables with **OrdinalEncoder** and **OneHotEncoder**.  
   - Scaled continuous features (e.g., `total_score`) using **StandardScaler**.
     
4. **Modeling & Evaluation**  
   - **Logistic Regression** (with and without class balancing + SMOTE).  
   - **CatBoost Classifier** (hyperparameter tuning via grid search).  
   - Used **Stratified K-Fold Cross-Validation** for robust evaluation.  

5. **Key Results**  
   - **CatBoost** achieved the **best recall for class 1 (Graduated)** and overall accuracy (~61%).  
   - Logistic Regression with SMOTE improved balance between classes but had lower recall for graduates.  

6. **Recommendations**  
   - Increase engagement for participants with low weekly commitment hours or test scores.  
   - Offer tailored support or prerequisite learning for those with limited prior experience.  
   - Strengthen communication channels where participants first hear about the program.  
   - Use predictive models early in the program to flag at-risk participants for additional mentorship.  

---

## Deployment  
The **Graduation Status Predictor** has been deployed on **Hugging Face Spaces** using **Gradio** for the interactive interface.  
You can try out the live demo here:  
👉 [Graduation Status Predictor on Hugging Face Spaces](https://huggingface.co/spaces/Decten/graduation-status-predictor)  

---

