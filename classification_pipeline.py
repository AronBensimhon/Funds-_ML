import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


# Preprocessing function adjusted for classification
def preprocess_for_classification(df):
    # Drop columns with excessive missing values or unnecessary for classification
    drop_features = [
        'AVG_DEPOSIT_FEE', 'ALPHA', 'YIELD_TRAILING_5_YRS',
        'AVG_ANNUAL_YIELD_TRAILING_5YRS', 'STANDARD_DEVIATION',
        'SHARPE_RATIO', 'YIELD_TRAILING_3_YRS', 'AVG_ANNUAL_YIELD_TRAILING_3YRS',
        'MANAGING_CORPORATION_LEGAL_ID', 'CONTROLLING_CORPORATION', 'FUND_NAME'
    ]
    df = df.drop(columns=drop_features)

    # Drop rows where target or essential features are missing
    df = df.dropna(subset=['FUND_CLASSIFICATION', 'MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD'], how='all')

    # Fill missing values in numerical features with median
    num_features = ['AVG_ANNUAL_MANAGEMENT_FEE', 'MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD']
    df[num_features] = df[num_features].fillna(df[num_features].median())

    # Convert categorical columns to numerical using Label Encoding
    categorical_features = ['FUND_CLASSIFICATION', 'TARGET_POPULATION', 'SPECIALIZATION', 'SUB_SPECIALIZATION',
                            'MANAGING_CORPORATION']
    encoders = {col: LabelEncoder() for col in categorical_features}
    for col, encoder in encoders.items():
        df[col] = encoder.fit_transform(df[col])

    # Convert date features to datetime format
    df['INCEPTION_DATE'] = pd.to_datetime(df['INCEPTION_DATE'], errors='coerce')
    df['REPORT_PERIOD'] = pd.to_datetime(df['REPORT_PERIOD'].astype(str), format='%Y%m')

    # Create a feature for the time difference between REPORT_PERIOD and INCEPTION_DATE
    df['TIME_ELAPSED'] = (df['REPORT_PERIOD'] - df['INCEPTION_DATE']).dt.days.fillna(0)

    # Replace NaN in non-critical date-derived features with a default value
    df['TIME_ELAPSED'] = df['TIME_ELAPSED'].fillna(0)

    # Drop original date columns after feature creation
    df = df.drop(columns=['INCEPTION_DATE', 'REPORT_PERIOD'])

    return df, encoders


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('gemel_net_dataset.csv')

    # Preprocess the dataset
    df_preprocessed, label_encoders = preprocess_for_classification(df)

    # Separate features and target variable
    X = df_preprocessed.drop(columns=['FUND_CLASSIFICATION'])
    y = df_preprocessed['FUND_CLASSIFICATION']

    # Impute any remaining missing values in the feature matrix
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Display feature importances
    feature_importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("Feature Importances:")
    print(importance_df)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in Random Forest Model')
    plt.gca().invert_yaxis()
    plt.show()

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Display results
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
