import pandas as pd
from datetime import timedelta


def preprocessing(df):
    # Identify Missing Values
    # print(indentify_missing_values(df))  # todo : uncomment

    # Remove specified features
    drop_features = [
        'AVG_DEPOSIT_FEE', 'ALPHA', 'YIELD_TRAILING_5_YRS',
        'AVG_ANNUAL_YIELD_TRAILING_5YRS', 'STANDARD_DEVIATION',
        'SHARPE_RATIO', 'YIELD_TRAILING_3_YRS', 'AVG_ANNUAL_YIELD_TRAILING_3YRS'
    ]
    df = df.drop(columns=drop_features)  # Avoid inplace

    # Remove rows where both 'MONTHLY_YIELD' and 'YEAR_TO_DATE_YIELD' are missing
    df = df.dropna(subset=['MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD'], how='all')

    # Checking again for missing values:
    # print(indentify_missing_values(df))  # todo : uncomment

    # Filling the NaN values with median:
    df[['AVG_ANNUAL_MANAGEMENT_FEE', 'MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD']] = df[
        ['AVG_ANNUAL_MANAGEMENT_FEE', 'MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD']
    ].fillna(df[['AVG_ANNUAL_MANAGEMENT_FEE', 'MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD']].median())

    # Convert features to the proper format for better use:
    # Convert 'INCEPTION_DATE' to datetime format
    df['INCEPTION_DATE'] = pd.to_datetime(df['INCEPTION_DATE'], errors='coerce')
    df['REPORT_PERIOD'] = pd.to_datetime(df['REPORT_PERIOD'].astype(str), format='%Y%m')

    # Redundant features:
    df = df.drop(columns=['MANAGING_CORPORATION_LEGAL_ID', 'CONTROLLING_CORPORATION'])

    # Creating new column DAYS_PASSED: from creation of קופה to report date
    # df['YEARS_PASSED'] = (df['REPORT_PERIOD'] - df['INCEPTION_DATE']).dt.days / 365
    # df['MONTHS_PASSED'] = (df['REPORT_PERIOD'] - df['INCEPTION_DATE']).dt.days / 30
    # df['DAYS_PASSED'] = (df['REPORT_PERIOD'] - df['INCEPTION_DATE']).dt.days
    df['TIME_ELAPSED'] = (df['REPORT_PERIOD'] - df['INCEPTION_DATE']).dt.days.apply(days_to_years_months_days)
    df = df.sort_values(by=['FUND_NAME', 'REPORT_PERIOD'], ascending=[True, True])

    return df


def indentify_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    print("Missing Values in Each Feature (Descending Order):")
    print(missing_values)
    return None


def days_to_years_months_days(days):
    # If days is NaN or NaT, return 'NaT'
    if pd.isna(days):
        return 'NaT'

    # Otherwise, calculate years, months, and remaining days
    years = days // 365
    months = (days % 365) // 30
    remaining_days = (days % 365) % 30
    return f"{years} years, {months} months, {remaining_days} days"


if __name__ == '__main__':
    df = pd.read_csv('gemel_net_dataset.csv')
    df_preprocessed = preprocessing(df)
    print(df_preprocessed)
