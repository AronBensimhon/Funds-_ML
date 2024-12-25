import pandas as pd


def preprocessing(df):
    # Identify Missing Values
    print(indentify_missing_values(df))

    # Remove specified features
    drop_features = [
        'AVG_DEPOSIT_FEE', 'ALPHA', 'YIELD_TRAILING_5_YRS',
        'AVG_ANNUAL_YIELD_TRAILING_5YRS', 'STANDARD_DEVIATION',
        'SHARPE_RATIO', 'YIELD_TRAILING_3_YRS', 'AVG_ANNUAL_YIELD_TRAILING_3YRS'
    ]
    df.drop(columns=drop_features, inplace=True)

    # Remove rows where both 'MONTHLY_YIELD' and 'YEAR_TO_DATE_YIELD' are missing
    df = df.dropna(subset=['MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD'], how='all')

    # Checking again for missing values:
    print(indentify_missing_values(df))

    # Filling the NaN values with median:
    df.fillna({
        'AVG_ANNUAL_MANAGEMENT_FEE': df['AVG_ANNUAL_MANAGEMENT_FEE'].median(),
        'MONTHLY_YIELD': df['MONTHLY_YIELD'].median(),
        'YEAR_TO_DATE_YIELD': df['YEAR_TO_DATE_YIELD'].median(),
        'AVG_DEPOSIT_FEE': df['AVG_DEPOSIT_FEE'].median(),
    }, inplace=True)
    return df


def indentify_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    print("Missing Values in Each Feature (Descending Order):")
    print(missing_values)
    return None


if __name__ == '__main__':
    df = pd.read_csv('gemel_net_dataset.csv')
    df_preprocessed = preprocessing(df)



