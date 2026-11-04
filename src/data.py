import pandas as pd

def load_credit_prediction(path):
    df = pd.read_csv(path)
    
    # Drop the old index column safely if it exists (handles raw vs cleaned files)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Fill missing values
    median_income = df['MonthlyIncome'].median()
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(median_income)
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
    
    # Remove rows where age = 0
    df = df[df['age'] > 0]
    
    # Replace placeholder codes (98 and 96) with 0 in past-due columns
    past_due_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse'
    ]
    for col in past_due_cols:
        df[col] = df[col].replace([98, 96], 0)
        
    # Return the cleaned dataframe
    return df