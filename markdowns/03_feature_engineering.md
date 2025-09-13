# 03 Feature Engineering

## a. Key Considerations from Exploratory Data Analysis


```python
import os

os.chdir("..")
print("Current working dir:", os.getcwd())
#print("Files in raw folder:", os.listdir("data/raw"))
```

    Current working dir: C:\Users\Window\Desktop\Everything_Data_Mentorship\mentorship_ds_project
    

## b. Imports 


```python
import pandas as pd

```

## c. Loading Data 


```python
df = pd.read_csv("data/interim/cleaned_df.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>age_range</th>
      <th>gender</th>
      <th>country</th>
      <th>referral_source</th>
      <th>years_experience</th>
      <th>track_applied</th>
      <th>weekly_commitment_hours</th>
      <th>main_aim</th>
      <th>motivation</th>
      <th>skill_level</th>
      <th>aptitude_test_completed</th>
      <th>total_score</th>
      <th>graduated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-12-01 23:50:47</td>
      <td>18-24 years</td>
      <td>Male</td>
      <td>Kenya</td>
      <td>Word of mouth</td>
      <td>Less than six months</td>
      <td>Data science</td>
      <td>less than 6 hours</td>
      <td>Upskill</td>
      <td>to enter into the data analysis career</td>
      <td>Beginner</td>
      <td>Yes</td>
      <td>58.67</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-12-03 09:35:19</td>
      <td>25-34 years</td>
      <td>Male</td>
      <td>Kenya</td>
      <td>WhatsApp</td>
      <td>6 months - 1 year</td>
      <td>Data science</td>
      <td>more than 14 hours</td>
      <td>Upskill</td>
      <td>To grow and improve my skills in data science ...</td>
      <td>Elementary</td>
      <td>Yes</td>
      <td>70.00</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-12-03 19:16:49</td>
      <td>18-24 years</td>
      <td>Female</td>
      <td>Kenya</td>
      <td>WhatsApp</td>
      <td>6 months - 1 year</td>
      <td>Data science</td>
      <td>more than 14 hours</td>
      <td>Upskill</td>
      <td>I’m motivated to join Everything Data to enhan...</td>
      <td>Intermediate</td>
      <td>Yes</td>
      <td>64.33</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-12-03 12:52:36</td>
      <td>18-24 years</td>
      <td>Female</td>
      <td>Kenya</td>
      <td>WhatsApp</td>
      <td>6 months - 1 year</td>
      <td>Data science</td>
      <td>7-14 hours</td>
      <td>Upskill</td>
      <td>I'd like to upskill and Join the Data Community</td>
      <td>Intermediate</td>
      <td>Yes</td>
      <td>75.00</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-12-03 18:12:27</td>
      <td>18-24 years</td>
      <td>Male</td>
      <td>Kenya</td>
      <td>WhatsApp</td>
      <td>Less than six months</td>
      <td>Data science</td>
      <td>7-14 hours</td>
      <td>Upskill</td>
      <td>I aim to join the mentorship program to enhanc...</td>
      <td>Beginner</td>
      <td>Yes</td>
      <td>59.00</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



## d. Dropping Unnecessary features

Due to class imbalance in the country column, I opted to drop the feature from the dataset.

Due to feature irrelevance, I dropped the timestamp column too.


```python
df = df.drop(["timestamp", "country", "motivation"], axis=1)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 115 entries, 0 to 114
    Data columns (total 11 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   age_range                115 non-null    object 
     1   gender                   115 non-null    object 
     2   referral_source          115 non-null    object 
     3   years_experience         115 non-null    object 
     4   track_applied            115 non-null    object 
     5   weekly_commitment_hours  115 non-null    object 
     6   main_aim                 115 non-null    object 
     7   skill_level              114 non-null    object 
     8   aptitude_test_completed  115 non-null    object 
     9   total_score              115 non-null    float64
     10  graduated                115 non-null    object 
    dtypes: float64(1), object(10)
    memory usage: 10.0+ KB
    

## e. Handle Missing Data


```python
df['skill_level'] = df['skill_level'].fillna(df['skill_level'].mode()[0])
```


```python
df['skill_level'].value_counts()
```




    skill_level
    Elementary      57
    Beginner        42
    Intermediate    16
    Name: count, dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 115 entries, 0 to 114
    Data columns (total 11 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   age_range                115 non-null    object 
     1   gender                   115 non-null    object 
     2   referral_source          115 non-null    object 
     3   years_experience         115 non-null    object 
     4   track_applied            115 non-null    object 
     5   weekly_commitment_hours  115 non-null    object 
     6   main_aim                 115 non-null    object 
     7   skill_level              115 non-null    object 
     8   aptitude_test_completed  115 non-null    object 
     9   total_score              115 non-null    float64
     10  graduated                115 non-null    object 
    dtypes: float64(1), object(10)
    memory usage: 10.0+ KB
    

## f. Correct data Types

I converted categorical/object columns to category dtype.


```python
# I created a function to wrap it all and ease the conversion
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
```


```python
df = set_ordered_categories(df)
```


```python
df['skill_level'].value_counts(dropna=False)
```




    skill_level
    Elementary      57
    Beginner        42
    Intermediate    16
    Name: count, dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 115 entries, 0 to 114
    Data columns (total 11 columns):
     #   Column                   Non-Null Count  Dtype   
    ---  ------                   --------------  -----   
     0   age_range                115 non-null    category
     1   gender                   115 non-null    object  
     2   referral_source          115 non-null    object  
     3   years_experience         115 non-null    category
     4   track_applied            115 non-null    object  
     5   weekly_commitment_hours  115 non-null    category
     6   main_aim                 115 non-null    object  
     7   skill_level              115 non-null    category
     8   aptitude_test_completed  115 non-null    object  
     9   total_score              115 non-null    float64 
     10  graduated                115 non-null    object  
    dtypes: category(4), float64(1), object(6)
    memory usage: 7.5+ KB
    

## g. For the target I'll apply binary encoding. 
This is the graduated column


```python
#Graduated column encoding 
df['graduated'] = df['graduated'].map({'No': 0, 'Yes': 1})
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_range</th>
      <th>gender</th>
      <th>referral_source</th>
      <th>years_experience</th>
      <th>track_applied</th>
      <th>weekly_commitment_hours</th>
      <th>main_aim</th>
      <th>skill_level</th>
      <th>aptitude_test_completed</th>
      <th>total_score</th>
      <th>graduated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18-24 years</td>
      <td>Male</td>
      <td>Word of mouth</td>
      <td>Less than six months</td>
      <td>Data science</td>
      <td>less than 6 hours</td>
      <td>Upskill</td>
      <td>Beginner</td>
      <td>Yes</td>
      <td>58.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25-34 years</td>
      <td>Male</td>
      <td>WhatsApp</td>
      <td>6 months - 1 year</td>
      <td>Data science</td>
      <td>more than 14 hours</td>
      <td>Upskill</td>
      <td>Elementary</td>
      <td>Yes</td>
      <td>70.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18-24 years</td>
      <td>Female</td>
      <td>WhatsApp</td>
      <td>6 months - 1 year</td>
      <td>Data science</td>
      <td>more than 14 hours</td>
      <td>Upskill</td>
      <td>Intermediate</td>
      <td>Yes</td>
      <td>64.33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18-24 years</td>
      <td>Female</td>
      <td>WhatsApp</td>
      <td>6 months - 1 year</td>
      <td>Data science</td>
      <td>7-14 hours</td>
      <td>Upskill</td>
      <td>Intermediate</td>
      <td>Yes</td>
      <td>75.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18-24 years</td>
      <td>Male</td>
      <td>WhatsApp</td>
      <td>Less than six months</td>
      <td>Data science</td>
      <td>7-14 hours</td>
      <td>Upskill</td>
      <td>Beginner</td>
      <td>Yes</td>
      <td>59.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 115 entries, 0 to 114
    Data columns (total 11 columns):
     #   Column                   Non-Null Count  Dtype   
    ---  ------                   --------------  -----   
     0   age_range                115 non-null    category
     1   gender                   115 non-null    object  
     2   referral_source          115 non-null    object  
     3   years_experience         115 non-null    category
     4   track_applied            115 non-null    object  
     5   weekly_commitment_hours  115 non-null    category
     6   main_aim                 115 non-null    object  
     7   skill_level              115 non-null    category
     8   aptitude_test_completed  115 non-null    object  
     9   total_score              115 non-null    float64 
     10  graduated                115 non-null    int64   
    dtypes: category(4), float64(1), int64(1), object(5)
    memory usage: 7.5+ KB
    


```python
df['main_aim'].value_counts(dropna=False)
```




    main_aim
    Upskill         74
    Learn afresh    23
    Portfolio       15
    Networking       2
    Other            1
    Name: count, dtype: int64



## h. Pefrom a train test split


```python
from sklearn.model_selection import train_test_split

#set the features and target variables
X = df.drop('graduated', axis=1)
y = df['graduated']

#Peform the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

## i. Encode Categorical Variables

Cosiderations

Ordinal Encoding for ordered categorical data. These columns are age_range, years_experience, weekly_commitment_hours and skill_level

For norminal features, we'll apply one hot encoding. These columns include; gender, referral_source, track_applied, main_aim, aptitude_test_completed




```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


#Identify column groups
ordinal_cols = ['age_range', 'years_experience', 'weekly_commitment_hours', 'skill_level']
nominal_cols = ['gender', 'referral_source', 'track_applied', 'main_aim', 'aptitude_test_completed']

#For OrdinalEncoder, I'll preserve the defined category order from the DataFrame
# I'll extract the categories directly from the categorical dtype
ordinal_categories = [X_train[col].cat.categories.tolist() for col in ordinal_cols]

# Build the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_cols)
    ],
    remainder='passthrough'  # Keep total_score without change
)

#Fit on training data only, transform both sets
X_train_final = preprocessor.fit_transform(X_train)
X_test_final  = preprocessor.transform(X_test)

# Optional: get feature names for inspection
oh_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(nominal_cols)
feature_names = ordinal_cols + list(oh_names) + ['total_score']
```


```python
oh_names
```




    array(['gender_Male', 'referral_source_Instagram',
           'referral_source_LinkedIn', 'referral_source_Twitter',
           'referral_source_WhatsApp', 'referral_source_Word of mouth',
           'referral_source_through a geeks for geeks webinar',
           'track_applied_Data science', 'main_aim_Networking',
           'main_aim_Other', 'main_aim_Portfolio', 'main_aim_Upskill',
           'aptitude_test_completed_Yes'], dtype=object)




```python
feature_names
```




    ['age_range',
     'years_experience',
     'weekly_commitment_hours',
     'skill_level',
     'gender_Male',
     'referral_source_Instagram',
     'referral_source_LinkedIn',
     'referral_source_Twitter',
     'referral_source_WhatsApp',
     'referral_source_Word of mouth',
     'referral_source_through a geeks for geeks webinar',
     'track_applied_Data science',
     'main_aim_Networking',
     'main_aim_Other',
     'main_aim_Portfolio',
     'main_aim_Upskill',
     'aptitude_test_completed_Yes',
     'total_score']



## j. Save the processor ready for use in modeling


```python
import joblib

# Save the fitted preprocessor
joblib.dump(preprocessor, 'artifacts/preprocessor.joblib')

# Saving the splits
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)
```
