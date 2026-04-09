"""
Description: Loads the Titanic dataset, performs data cleaning, exploratory analysis,
             and derives survival insights using pandas groupby and filtering.
"""

import pandas as pd

# ============================
# CONFIGURATION
# ============================
FILE_PATH = "Titanic-Dataset.csv"

# ============================
# DATA LOADING & EXPLORATION
# ============================
def load_data(filepath: str) -> pd.DataFrame:
    """Loads Titanic dataset from CSV."""
    return pd.read_csv(filepath)

def explore_data(df: pd.DataFrame) -> None:
    """Prints head, info, and describe for initial exploration."""
    print("\n" + "=" * 70)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print("\n--- First 5 Rows ---")
    print(df.head())
    print("\n--- Dataset Info ---")
    print(df.info())
    print("\n--- Statistical Summary ---")
    print(df.describe())

# ============================
# DATA CLEANING
# ============================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values and removes unnecessary columns.
    
    - Age: Fill missing with median.
    - Embarked: Fill missing with mode.
    - Cabin: Drop column (too many missing values).
    - Duplicates: Remove any duplicate rows.
    """
    df_clean = df.copy()
    
    # Fill missing Age with median
    median_age = df_clean["Age"].median()
    df_clean["Age"].fillna(median_age, inplace=True)
    
    # Fill missing Embarked with mode (most frequent port)
    mode_embarked = df_clean["Embarked"].mode()[0]
    df_clean["Embarked"].fillna(mode_embarked, inplace=True)
    
    # Drop Cabin column
    df_clean.drop("Cabin", axis=1, inplace=True)
    
    # Remove duplicate rows
    df_clean.drop_duplicates(inplace=True)
    
    return df_clean

def verify_cleaning(df: pd.DataFrame) -> None:
    """Prints missing value counts after cleaning."""
    print("\n" + "=" * 70)
    print("STEP 2: DATA CLEANING VERIFICATION")
    print("=" * 70)
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing[missing > 0] if missing.any() else "No missing values.")

# ============================
# DATA ANALYSIS (GroupBy)
# ============================
def analyze_survival(df: pd.DataFrame) -> None:
    """Performs groupby analyses: survival by gender, class, age group, and average age per class."""
    print("\n" + "=" * 70)
    print("STEP 3: DATA ANALYSIS (GROUPBY)")
    print("=" * 70)
    
    # Survival rate by gender
    print("\n--- Survival Rate by Gender ---")
    print(df.groupby("Sex")["Survived"].mean().map("{:.2%}".format))
    
    # Survival rate by passenger class
    print("\n--- Survival Rate by Passenger Class ---")
    print(df.groupby("Pclass")["Survived"].mean().map("{:.2%}".format))
    
    # Average age per class
    print("\n--- Average Age per Class ---")
    print(df.groupby("Pclass")["Age"].mean().round(1))
    
    # Create age groups
    bins = [0, 12, 18, 35, 60, 100]
    labels = ["Child (0-12)", "Teen (13-18)", "Young Adult (19-35)", "Adult (36-60)", "Senior (60+)"]
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    
    print("\n--- Survival Rate by Age Group ---")
    survival_by_age = df.groupby("Age_Group", observed=False)["Survived"].mean().map("{:.2%}".format)
    print(survival_by_age)

# ============================
# FILTERING
# ============================
def filter_passengers(df: pd.DataFrame) -> None:
    """Extracts and displays specific subsets: female survivors, child survivors, 1st class survivors."""
    print("\n" + "=" * 70)
    print("STEP 4: FILTERING")
    print("=" * 70)
    
    # Female survivors
    female_survived = df[(df["Sex"] == "female") & (df["Survived"] == 1)]
    print(f"\n--- Female Survivors (total: {len(female_survived)}) ---")
    print(female_survived[["Name", "Pclass", "Age", "Embarked"]].head())
    
    # Child survivors (age < 18)
    children_survived = df[(df["Age"] < 18) & (df["Survived"] == 1)]
    print(f"\n--- Child Survivors (total: {len(children_survived)}) ---")
    print(children_survived[["Name", "Sex", "Pclass", "Age"]].head())
    
    # 1st class survivors
    first_class_survivors = df[(df["Pclass"] == 1) & (df["Survived"] == 1)]
    print(f"\n--- 1st Class Survivors (total: {len(first_class_survivors)}) ---")
    print(first_class_survivors[["Name", "Sex", "Age", "Embarked"]].head())

# ============================
# INSIGHTS
# ============================
def generate_insights(df: pd.DataFrame) -> None:
    """Calculates and prints key survival insights based on the analysis."""
    print("\n" + "=" * 70)
    print("STEP 5: KEY INSIGHTS")
    print("=" * 70)
    
    # Gender effect
    female_rate = df[df["Sex"] == "female"]["Survived"].mean()
    male_rate = df[df["Sex"] == "male"]["Survived"].mean()
    print(f"\n1. Gender Impact:\n   Women had a significantly higher survival rate ({female_rate:.1%}) compared to men ({male_rate:.1%}).")
    
    # Class effect
    class1_rate = df[df["Pclass"] == 1]["Survived"].mean()
    class3_rate = df[df["Pclass"] == 3]["Survived"].mean()
    print(f"\n2. Socioeconomic Status:\n   1st class passengers survived at a rate of {class1_rate:.1%}, while 3rd class survived at only {class3_rate:.1%}.")
    
    # Children prioritization
    child_rate = df[df["Age"] < 18]["Survived"].mean()
    adult_rate = df[df["Age"] >= 18]["Survived"].mean()
    print(f"\n3. Age Prioritization:\n   Children (under 18) had a survival rate of {child_rate:.1%}, higher than adults ({adult_rate:.1%}).")
    
    # Best combination
    best_group = df.groupby(["Sex", "Pclass", "Age_Group"], observed=False)["Survived"].mean().idxmax()
    best_rate = df.groupby(["Sex", "Pclass", "Age_Group"], observed=False)["Survived"].mean().max()
    print(f"\n4. Highest Survival Profile:\n   The group with the highest survival rate ({best_rate:.1%}) was: {best_group}.")
    
    print("\n" + "=" * 70)
    print("CONCLUSION: Women, first-class passengers, and children were prioritized during evacuation.")
    print("=" * 70)

# ============================
# MAIN EXECUTION PIPELINE
# ============================
def main():
    """Runs the complete Titanic analysis pipeline."""
    # Load data
    df_raw = load_data(FILE_PATH)
    
    # Explore
    explore_data(df_raw)
    
    # Clean
    df_clean = clean_data(df_raw)
    verify_cleaning(df_clean)
    
    # Analyze
    analyze_survival(df_clean)
    
    # Filter
    filter_passengers(df_clean)
    
    # Insights
    generate_insights(df_clean)

if __name__ == "__main__":
    main()
