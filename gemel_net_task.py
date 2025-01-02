import pandas as pd


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

    # Filling the NaN values with median for relevant features:
    df[['AVG_ANNUAL_MANAGEMENT_FEE', 'MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD']] = df[
        ['AVG_ANNUAL_MANAGEMENT_FEE', 'MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD']
    ].fillna(df[['AVG_ANNUAL_MANAGEMENT_FEE', 'MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD']].median())

    # Convert features to the proper format for better use:
    # Convert 'INCEPTION_DATE' to datetime format
    df['INCEPTION_DATE'] = pd.to_datetime(df['INCEPTION_DATE'], errors='coerce')
    df['REPORT_PERIOD'] = pd.to_datetime(df['REPORT_PERIOD'].astype(str), format='%Y%m')

    # Filter rows where REPORT_PERIOD is on or after January 1, 2000
    df = df[df['REPORT_PERIOD'] >= '2020-01-01']
    # Redundant features:
    df = df.drop(columns=['MANAGING_CORPORATION_LEGAL_ID', 'CONTROLLING_CORPORATION'])

    # Creating new column TIME_ELAPSED: from creation of קופה to report date
    df['TIME_ELAPSED'] = (df['REPORT_PERIOD'] - df['INCEPTION_DATE']).dt.days.apply(days_to_years_months_days)
    df = df.sort_values(by=['FUND_NAME', 'REPORT_PERIOD'], ascending=[True, True])

    # Group by MANAGING_CORPORATION and REPORT_PERIOD and sum specified features
    group_sum_features = ['DEPOSITS', 'WITHDRAWLS', 'INTERNAL_TRANSFERS', 'NET_MONTHLY_DEPOSITS']
    df_grouped = df.groupby(['FUND_NAME'])[group_sum_features].sum().round(2).reset_index()
    df_grouped_managing_corp = df.groupby(['MANAGING_CORPORATION'])[group_sum_features].sum().round(2).reset_index()

    features_to_percent= ['STOCK_MARKET_EXPOSURE', 'FOREIGN_EXPOSURE',	'FOREIGN_CURRENCY_EXPOSURE']
    for feature in features_to_percent:
        df[f"{feature}_%"] = df[feature] / df['TOTAL_ASSETS']
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
    years = days // 365
    months = (days % 365) // 30
    remaining_days = (days % 365) % 30
    return f"{years} years, {months} months, {remaining_days} days"


if __name__ == '__main__':
    df = pd.read_csv('gemel_net_dataset.csv')
    df_preprocessed = preprocessing(df)
    print(df_preprocessed)

