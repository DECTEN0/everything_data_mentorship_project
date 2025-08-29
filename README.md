# Everything Data – Mentorship Cohort Analysis (Data Science Track)

This repository analyzes a mentorship cohort dataset to understand demographics, motivations, performance, and drivers of graduation.

## Dataset Columns
- Timestamp
- ID No.
- Age range
- Gender
- Country
- Where did you hear about Everything Data?
- Years of learning experience in the data field
- Track applied for
- Hours per week available
- Main aim for joining
- Motivation for joining
- Self-assessed skill level in chosen track
- Aptitude test completion status
- Total score
- Graduation status

## Project Goals
1) Understand participant demographics, motivations, and performance  
2) Highlight factors that influence graduation rates  
3) Present actionable recommendations for improving future cohorts

## Quickstart
```bash
# 1) Create a virtual environment (example with venv)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Put the provided CSV into data/raw/ as cohort.csv

# 4) Run the pipeline
python scripts/run_all.py --config configs/params.yaml

# 5) Explore outputs in reports/ and notebooks/
```

## Repo Layout
See the folder tree in this README or the repository itself.
