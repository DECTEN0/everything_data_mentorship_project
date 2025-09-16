import pandas as pd
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