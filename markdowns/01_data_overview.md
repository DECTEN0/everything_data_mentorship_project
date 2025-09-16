# 01 data overview

The goal of this project is to use classification models to predict student graduation status and to identify the key factors influencing graduation. By evaluating different classification models and analyzing feature importance, the project aims to provide actionable recommendations to help institutions improve graduation rates and better support at-risk students.


```python

import pandas as pd

```


```python
df1 = pd.read_csv("data/raw/Cohort_3.csv")
```


```python
df2 = pd.read_csv("data/raw/Cohort_3_DA.csv")
```


```python
df = pd.concat([df1, df2])
df.tail()
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
      <th>Timestamp</th>
      <th>Id. No</th>
      <th>Age range</th>
      <th>Gender</th>
      <th>Country</th>
      <th>Where did you hear about Everything Data?</th>
      <th>How many years of learning experience do you have in the field of data?</th>
      <th>Which track are you applying for?</th>
      <th>How many hours per week can you commit to learning?</th>
      <th>What is your main aim for joining the mentorship program?</th>
      <th>What is your motivation to join the Everything Data mentorship program?</th>
      <th>How best would you describe your skill level in the track you are applying for?</th>
      <th>Have you completed the everything data aptitude test for your track?</th>
      <th>Total score</th>
      <th>Graduated</th>
      <th>ID No.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>12/2/2024 21:06:50</td>
      <td>NaN</td>
      <td>25-34 years</td>
      <td>Male</td>
      <td>Kenya</td>
      <td>WhatsApp</td>
      <td>Less than six months</td>
      <td>Data analysis</td>
      <td>7-14 hours</td>
      <td>Upskill</td>
      <td>My motivation to join the Everything Data ment...</td>
      <td>Elementary - I have theoretical understanding ...</td>
      <td>No</td>
      <td>62.33</td>
      <td>No</td>
      <td>DA348</td>
    </tr>
    <tr>
      <th>48</th>
      <td>12/3/2024 20:11:37</td>
      <td>NaN</td>
      <td>25-34 years</td>
      <td>Male</td>
      <td>Kenya</td>
      <td>LinkedIn</td>
      <td>Less than six months</td>
      <td>Data analysis</td>
      <td>7-14 hours</td>
      <td>Build a project portfolio</td>
      <td>Heavy inspiration from the collaboration and i...</td>
      <td>Elementary - I have theoretical understanding ...</td>
      <td>Yes</td>
      <td>70.67</td>
      <td>No</td>
      <td>DA349</td>
    </tr>
    <tr>
      <th>49</th>
      <td>11/28/2024 15:18:09</td>
      <td>NaN</td>
      <td>25-34 years</td>
      <td>Male</td>
      <td>Kenya</td>
      <td>WhatsApp</td>
      <td>6 months - 1 year</td>
      <td>Data analysis</td>
      <td>more than 14 hours</td>
      <td>Learn data afresh</td>
      <td>I am interested in building my data skills so ...</td>
      <td>Elementary - I have theoretical understanding ...</td>
      <td>Yes</td>
      <td>70.67</td>
      <td>No</td>
      <td>DA350</td>
    </tr>
    <tr>
      <th>50</th>
      <td>11/29/2024 16:49:09</td>
      <td>NaN</td>
      <td>25-34 years</td>
      <td>Female</td>
      <td>Kenya</td>
      <td>WhatsApp</td>
      <td>Less than six months</td>
      <td>Data analysis</td>
      <td>more than 14 hours</td>
      <td>Build a project portfolio</td>
      <td>I’m eager to join the the mentorship program t...</td>
      <td>Elementary - I have theoretical understanding ...</td>
      <td>Yes</td>
      <td>65.67</td>
      <td>Yes</td>
      <td>DA351</td>
    </tr>
    <tr>
      <th>51</th>
      <td>12/1/2024 0:11:49</td>
      <td>NaN</td>
      <td>25-34 years</td>
      <td>Male</td>
      <td>Kenya</td>
      <td>Twitter</td>
      <td>6 months - 1 year</td>
      <td>Data analysis</td>
      <td>7-14 hours</td>
      <td>Upskill</td>
      <td>I developed an interest in data after graduati...</td>
      <td>Intermediate - I have theoretical knowledge an...</td>
      <td>Yes</td>
      <td>70.67</td>
      <td>No</td>
      <td>DA352</td>
    </tr>
  </tbody>
</table>
</div>



1. <b>Schema Checkes; data types


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 115 entries, 0 to 51
    Data columns (total 16 columns):
     #   Column                                                                           Non-Null Count  Dtype  
    ---  ------                                                                           --------------  -----  
     0   Timestamp                                                                        115 non-null    object 
     1   Id. No                                                                           63 non-null     object 
     2   Age range                                                                        115 non-null    object 
     3   Gender                                                                           115 non-null    object 
     4   Country                                                                          115 non-null    object 
     5   Where did you hear about Everything Data?                                        115 non-null    object 
     6   How many years of learning experience do you have in the field of data?          115 non-null    object 
     7   Which track are you applying for?                                                115 non-null    object 
     8   How many hours per week can you commit to learning?                              115 non-null    object 
     9   What is your main aim for joining the mentorship program?                        115 non-null    object 
     10  What is your motivation to join the Everything Data mentorship program?          115 non-null    object 
     11  How best would you describe your skill level in the track you are applying for?  115 non-null    object 
     12  Have you completed the everything data aptitude test for your track?             115 non-null    object 
     13  Total score                                                                      115 non-null    float64
     14  Graduated                                                                        115 non-null    object 
     15  ID No.                                                                           52 non-null     object 
    dtypes: float64(1), object(15)
    memory usage: 15.3+ KB
    


```python
#drop the id no columns
df.drop(columns=['Id. No', 'ID No.'], inplace=True)
```


```python
df.head(2)
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
      <th>Timestamp</th>
      <th>Age range</th>
      <th>Gender</th>
      <th>Country</th>
      <th>Where did you hear about Everything Data?</th>
      <th>How many years of learning experience do you have in the field of data?</th>
      <th>Which track are you applying for?</th>
      <th>How many hours per week can you commit to learning?</th>
      <th>What is your main aim for joining the mentorship program?</th>
      <th>What is your motivation to join the Everything Data mentorship program?</th>
      <th>How best would you describe your skill level in the track you are applying for?</th>
      <th>Have you completed the everything data aptitude test for your track?</th>
      <th>Total score</th>
      <th>Graduated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12/1/2024 23:50:47</td>
      <td>18-24 years</td>
      <td>Male</td>
      <td>Kenya</td>
      <td>Word of mouth</td>
      <td>Less than six months</td>
      <td>Data science</td>
      <td>less than 6 hours</td>
      <td>Upskill</td>
      <td>to enter into the data analysis career</td>
      <td>Beginner - I have NO learning or work experien...</td>
      <td>Yes</td>
      <td>58.67</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12/3/2024 9:35:19</td>
      <td>25-34 years</td>
      <td>Male</td>
      <td>Kenya</td>
      <td>WhatsApp</td>
      <td>6 months - 1 year</td>
      <td>Data science</td>
      <td>more than 14 hours</td>
      <td>Upskill</td>
      <td>To grow and improve my skills in data science ...</td>
      <td>Elementary - I have theoretical understanding ...</td>
      <td>Yes</td>
      <td>70.00</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Covert Timestamp to datetime from a string
import datetime as dt
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.dtypes
```




    Timestamp                                                                          datetime64[ns]
    Age range                                                                                  object
    Gender                                                                                     object
    Country                                                                                    object
    Where did you hear about Everything Data?                                                  object
    How many years of learning experience do you have in the field of data?                    object
    Which track are you applying for?                                                          object
    How many hours per week can you commit to learning?                                        object
    What is your main aim for joining the mentorship program?                                  object
    What is your motivation to join the Everything Data mentorship program?                    object
    How best would you describe your skill level in the track you are applying for?            object
    Have you completed the everything data aptitude test for your track?                       object
    Total score                                                                               float64
    Graduated                                                                                  object
    dtype: object




```python
df.head(2)
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
      <th>Timestamp</th>
      <th>Age range</th>
      <th>Gender</th>
      <th>Country</th>
      <th>Where did you hear about Everything Data?</th>
      <th>How many years of learning experience do you have in the field of data?</th>
      <th>Which track are you applying for?</th>
      <th>How many hours per week can you commit to learning?</th>
      <th>What is your main aim for joining the mentorship program?</th>
      <th>What is your motivation to join the Everything Data mentorship program?</th>
      <th>How best would you describe your skill level in the track you are applying for?</th>
      <th>Have you completed the everything data aptitude test for your track?</th>
      <th>Total score</th>
      <th>Graduated</th>
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
      <td>Beginner - I have NO learning or work experien...</td>
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
      <td>Elementary - I have theoretical understanding ...</td>
      <td>Yes</td>
      <td>70.00</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



2. <b>Renaming the columns


```python
df.columns
```




    Index(['Timestamp', 'Age range', 'Gender', 'Country',
           'Where did you hear about Everything Data?',
           'How many years of learning experience do you have in the field of data?',
           'Which track are you applying for?',
           'How many hours per week can you commit to learning?',
           'What is your main aim for joining the mentorship program?',
           'What is your motivation to join the Everything Data mentorship program?',
           'How best would you describe your skill level in the track you are applying for?',
           'Have you completed the everything data aptitude test for your track?',
           'Total score', 'Graduated'],
          dtype='object')




```python
new_columns = [
    "timestamp",
    "age_range",
    "gender",
    "country",
    "referral_source",
    "years_experience",
    "track_applied",
    "weekly_commitment_hours",
    "main_aim",
    "motivation",
    "skill_level",
    "aptitude_test_completed",
    "total_score",
    "graduated"
]
df.columns = new_columns
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
      <td>Beginner - I have NO learning or work experien...</td>
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
      <td>Elementary - I have theoretical understanding ...</td>
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
      <td>Intermediate - I have theoretical knowledge an...</td>
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
      <td>Intermediate - I have theoretical knowledge an...</td>
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
      <td>Beginner - I have NO learning or work experien...</td>
      <td>Yes</td>
      <td>59.00</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



3. <b>Categorical Standardization


```python
#Keeep the years_experience ordered and categorical for visualization
experience_categories = [
    "Less than six months",
    "6 months - 1 year",
    "1-3 years",
    "4-6 years"
]

df["years_experience"] = pd.Categorical(
    df["years_experience"], 
    categories=experience_categories, 
    ordered=True
)
```


```python
#ordered categories
commitment_categories = ["less than 6 hours", "7-14 hours", "more than 14 hours"]

df["weekly_commitment_hours"] = pd.Categorical(
    df["weekly_commitment_hours"], 
    categories=commitment_categories, 
    ordered=True
)
```

We have overlaps like "Upskill" and "both upskilling and connecting with fellow data professionals".
To make analysis meaningful, I could consolidate:

Upskill → includes "Upskill" and "both upskilling and connecting..."

Learn afresh → "Learn data afresh"

Portfolio → "Build a project portfolio"

Networking → "Connect with fellow data professionals"


```python

def clean_main_aim(x):
    if "upskill" in x.lower():
        return "Upskill"
    elif "afresh" in x.lower():
        return "Learn afresh"
    elif "portfolio" in x.lower():
        return "Portfolio"
    elif "connect" in x.lower():
        return "Networking"
    else:
        return "Other"

df["main_aim"] = df["main_aim"].apply(clean_main_aim)
df["main_aim"].value_counts()
```




    main_aim
    Upskill         74
    Learn afresh    23
    Portfolio       15
    Networking       2
    Other            1
    Name: count, dtype: int64



The raw labels are long and slightly overlapping and unordered. A cleaned version might look like:

Beginner → "Beginner - I have NO learning or work experience"

Elementary → "Elementary - I have theoretical understanding of basics"

Intermediate → "Intermediate - I have theoretical knowledge + experience"


```python
skill_level_categories = ["Beginner", "Elementary", "Intermediate"]

df["skill_level"] = pd.Categorical(
    df["skill_level"].replace({
        "Beginner - I have NO learning or work experience in data analysis/ data science": "Beginner",
        "Elementary - I have theoretical understanding of basic data analysis/ data science concepts": "Elementary",
        "Intermediate - I have theoretical knowledge and experience in data analysis/ data science": "Intermediate"
    }),
    categories=skill_level_categories,
    ordered=True
)
df["skill_level"].value_counts()
```




    skill_level
    Elementary      56
    Beginner        42
    Intermediate    16
    Name: count, dtype: int64



We can kee age ranges are ordinal categories, so the goal is to represent them in a way that shows the distribution across age groups clearly.


```python
age_categories = ["18-24 years", "25-34 years", "35-44 years", "45-54 years"]

df["age_range"] = pd.Categorical(df["age_range"], categories=age_categories, ordered=True)
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




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 115 entries, 0 to 51
    Data columns (total 14 columns):
     #   Column                   Non-Null Count  Dtype         
    ---  ------                   --------------  -----         
     0   timestamp                115 non-null    datetime64[ns]
     1   age_range                115 non-null    category      
     2   gender                   115 non-null    object        
     3   country                  115 non-null    object        
     4   referral_source          115 non-null    object        
     5   years_experience         115 non-null    category      
     6   track_applied            115 non-null    object        
     7   weekly_commitment_hours  115 non-null    category      
     8   main_aim                 115 non-null    object        
     9   motivation               115 non-null    object        
     10  skill_level              114 non-null    category      
     11  aptitude_test_completed  115 non-null    object        
     12  total_score              115 non-null    float64       
     13  graduated                115 non-null    object        
    dtypes: category(4), datetime64[ns](1), float64(1), object(8)
    memory usage: 11.0+ KB
    

Save the cleaned data ready for Exploratory data analysis


```python
df.to_csv("data/interim/cleaned_df.csv", index=False)
```


```python

```
