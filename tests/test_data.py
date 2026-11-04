import pandas as pd
from src.data import load_credit_prediction

def test_load_credit_prediction(tmp_path):
    # Create a mock raw dataframe with edge cases to test your pipeline
    data = {
        'Unnamed: 0': [1, 2, 2, 3], # Duplicate row simulation
        'SeriousDlqin2yrs': [0, 1, 1, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.1, 0.5, 0.5, 0.2],
        'age': [30, 0, 0, 45], # age=0 to test row removal
        'NumberOfTime30-59DaysPastDueNotWorse': [0, 98, 98, 1], # Placeholder 98
        'DebtRatio': [0.2, 0.3, 0.3, 0.4],
        'MonthlyIncome': [5000, None, None, 7000], # Missing values for median fill
        'NumberOfOpenCreditLinesAndLoans': [5, 3, 3, 4],
        'NumberOfTimes90DaysLate': [0, 0, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 0, 0, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 96, 96, 0], # Placeholder 96
        'NumberOfDependents': [1, None, None, 2] # Missing dependents
    }
    
    df_mock = pd.DataFrame(data)
    
    # 2. Write it to a temporary CSV file using pytest's tmp_path fixture
    csv_file = tmp_path / 'test_credit.csv'
    df_mock.to_csv(csv_file, index=False)
    
    # 3. Run your cleaning function on the mock file
    result_df = load_credit_prediction(csv_file)
    
    # 4. Assertions to verify everything worked correctly
    # - 'Unnamed: 0' index column should be dropped
    assert 'Unnamed: 0' not in result_df.columns
    
    # - Rows with age = 0 should be gone, and duplicates dropped
    assert (result_df['age'] > 0).all()
    
    # - Placeholders (98 and 96) should be replaced with 0
    assert result_df['NumberOfTime30-59DaysPastDueNotWorse'].max() <= 1
    assert result_df['NumberOfTime60-89DaysPastDueNotWorse'].max() == 0
    
    # - Missing values should be successfully filled (no NaNs left)
    assert result_df['MonthlyIncome'].isna().sum() == 0
    assert result_df['NumberOfDependents'].isna().sum() == 0