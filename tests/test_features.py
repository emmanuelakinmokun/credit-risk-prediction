import pandas as pd
import numpy as np
from src.features import load_dataset

def test_load_dataset(tmp_path):
    # Create dummy input data
    data = {
        'MonthlyIncome': [5000.0, 0.0, 12000.0],
        'RevolvingUtilizationOfUnsecuredLines': [0.4, 1.2, 0.0],
        'DebtRatio': [0.25, 1.8, 0.05],
        'NumberOfDependents': [0, 2, 1],
        'NumberOfOpenCreditLinesAndLoans': [6, 10, 4],
        'NumberRealEstateLoansOrLines': [1, 2, 0]
    }

    df_input = pd.DataFrame(data)

    # Write to a temporary csv file for testing
    csv_file = tmp_path/'sample_data.csv'
    df_input.to_csv(csv_file, index=False)

    # Run the load_dataset function
    df_result = load_dataset(str(csv_file))

    # Assertions
    assert isinstance(df_result, pd.DataFrame)
    assert len(df_result) == 3

    # Check that all new feature columns were successfully added
    expected_columns = [
        'Log_MonthlyIncome',
        'Log_Utilization',
        'Log_DebtRatio',
        'IncomePerDependent',
        'IsHighUtilization',
        'UnsecuredCreditLines'
    ]
    for col in expected_columns:
        assert col in df_result

    # Verify log transformation calculations
    np.testing.assert_allclose(df_result['Log_MonthlyIncome'].iloc[0], np.log1p(5000.0))
    np.testing.assert_allclose(df_result['Log_Utilization'].iloc[1], np.log1p(1.2))

    # Verify IsHighUtilization flag (> 1.0 threshold)
    assert df_result['IsHighUtilization'].tolist() == [0, 1, 0]

    # Verify UnsecuredCreditLines calculation
    assert df_result['UnsecuredCreditLines'].tolist() == [5, 8, 4]