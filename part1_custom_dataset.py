"""
Description: Creates a pandas DataFrame from a dictionary with 5 columns, 15 rows,
             and a custom index (Employee_ID). Demonstrates DataFrame construction
             and basic data inspection.
"""

import pandas as pd

def create_custom_dataset() -> pd.DataFrame:
    """
    Builds a DataFrame containing employee performance records.
    
    Returns:
        pd.DataFrame: Indexed by Employee_ID with columns:
                      Name, Department, Salary, Experience_Years, Performance_Score.
    """
    # Dictionary with 5 features and 15 observations
    data = {
        "Name": [
            "Abebe", "kebede", "chala", "gebeyew", "demsis",
            "demelash", "gemechu", "abel", "hagos", "belay",
            "habtamu", "abera", "grma", "abekal", "asgdom"
        ],
        "Department": [
            "HR", "IT", "IT", "Marketing", "HR",
            "IT", "Marketing", "Finance", "Finance", "IT",
            "HR", "Marketing", "Finance", "IT", "HR"
        ],
        "Salary": [
            55000, 72000, 68000, 58000, 53000,
            75000, 61000, 82000, 79000, 71000,
            54000, 59000, 81000, 69500, 56000
        ],
        "Experience_Years": [
            3, 7, 5, 4, 2,
            8, 6, 10, 9, 6,
            3, 4, 11, 7, 2
        ],
        "Performance_Score": [
            85, 92, 88, 79, 91,
            95, 84, 96, 94, 89,
            87, 82, 98, 90, 88
        ]
    }
    
    # Create DataFrame and set custom index
    df = pd.DataFrame(data)
    df.index = range(101, 116)  # Employee_ID from 101 to 115
    df.index.name = "Employee_ID"
    
    return df

def main():
    """Executes the custom dataset creation and displays results."""
    df_employees = create_custom_dataset()
    
    print("=" * 60)
    print("CUSTOM DATASET: EMPLOYEE PERFORMANCE RECORDS")
    print("=" * 60)
    print(df_employees)
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(df_employees.info())
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(df_employees.describe())

if __name__ == "__main__":
    main()
