import pandas as pd
import numpy as np

def load_dataset(path):
    df = pd.read_csv(path)
    # Apply log transform to tame extreme skewness
    df['Log_MonthlyIncome'] = np.log1p(df['MonthlyIncome'])
    df['Log_Utilization'] = np.log1p(df['RevolvingUtilizationOfUnsecuredLines'])
    df['Log_DebtRatio'] = np.log1p(df['DebtRatio'])

    # Add 1 to dependents to avoid division-by-zero errors when someone has 0 dependents
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    # Flag those with maxed out cards
    df['IsHighUtilization'] = (df['RevolvingUtilizationOfUnsecuredLines'] > 1.0).astype(int)
    # Avoid division by zero by using safety checks or handling zeros
    df['UnsecuredCreditLines'] = df['NumberOfOpenCreditLinesAndLoans'] - df['NumberRealEstateLoansOrLines']

    return df



