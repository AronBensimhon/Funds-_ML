import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def preprocessing(df):
    drop_features = [  # Columns with excessive missing values or unnecessary for classification
        'AVG_DEPOSIT_FEE', 'ALPHA', 'YIELD_TRAILING_5_YRS',
        'AVG_ANNUAL_YIELD_TRAILING_5YRS', 'STANDARD_DEVIATION',
        'SHARPE_RATIO', 'YIELD_TRAILING_3_YRS', 'AVG_ANNUAL_YIELD_TRAILING_3YRS',
        'MANAGING_CORPORATION_LEGAL_ID', 'CONTROLLING_CORPORATION', 'FUND_NAME', 'FUND_ID'
    ]
    df = df.drop(columns=drop_features)  # Drop selected columns

    df = df.dropna(subset=['MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD'], how='all')  # Drop rows with missing essential features

    num_features = ['AVG_ANNUAL_MANAGEMENT_FEE', 'MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD']  # Numerical features
    df[num_features] = df[num_features].fillna(df[num_features].median())  # Fill missing values with median

    df['INCEPTION_DATE'] = pd.to_datetime(df['INCEPTION_DATE'], errors='coerce')  # Convert to datetime
    df['REPORT_PERIOD'] = pd.to_datetime(df['REPORT_PERIOD'].astype(str), format='%Y%m')  # Convert to datetime
    df['TIME_ELAPSED_YEARS'] = ((df['REPORT_PERIOD'] - df['INCEPTION_DATE']).dt.days.fillna(0)) / 365.25  # Calculate elapsed years
    df['TIME_ELAPSED_YEARS'] = df['TIME_ELAPSED_YEARS'].fillna(0)  # Fill missing values with 0

    df = df.drop(columns=['INCEPTION_DATE', 'REPORT_PERIOD'])  # Drop original date columns

    df['FEE_BUCKET'] = pd.cut(  # Bucketize AVG_ANNUAL_MANAGEMENT_FEE
        df['AVG_ANNUAL_MANAGEMENT_FEE'],
        bins=[-1, 0.5, 1.5, float('inf')],
        labels=['Low Fee', 'Medium Fee', 'High Fee']
    )

    df['TOTAL_FEES'] = df['AVG_ANNUAL_MANAGEMENT_FEE'] * df['TOTAL_ASSETS']  # Calculate total fees collected

    categorical_features = ['FUND_CLASSIFICATION', 'TARGET_POPULATION', 'SPECIALIZATION', 'SUB_SPECIALIZATION',
                            'MANAGING_CORPORATION', 'FEE_BUCKET']  # Categorical features

    encoders = {col: LabelEncoder() for col in categorical_features}  # Create LabelEncoder objects
    for col, encoder in encoders.items():
        df[col] = encoder.fit_transform(df[col])  # Encode categorical features

    return df, encoders


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on test data

    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    f1 = f1_score(y_test, y_pred, average='weighted')  # Calculate F1 score
    report = classification_report(y_test, y_pred)  # Generate classification report

    cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')  # Cross-validation
    cv_mean = cross_val_scores.mean()  # Mean cross-validation score
    return {
        "model": model,
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": report,
        "cross_val_mean": cv_mean
    }


def plot_comparison(results):
    model_names = list(results.keys())  # Extract model names
    accuracies = [metrics['accuracy'] for metrics in results.values()]  # Extract accuracies
    f1_scores = [metrics['f1_score'] for metrics in results.values()]  # Extract F1 scores

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, alpha=0.7, label='Accuracy', width=0.4, align='center')  # Plot accuracies
    plt.bar(model_names, f1_scores, alpha=0.7, label='F1-Score', width=0.4, align='edge')  # Plot F1 scores
    plt.xlabel('Model')  # X-axis label
    plt.ylabel('Score')  # Y-axis label
    plt.title('Model Performance Comparison')  # Plot title
    plt.legend()  # Add legend
    plt.tight_layout()
    plt.show()


def display_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, "feature_importances_"):  # Check if feature importances are supported
        feature_importances = model.feature_importances_  # Get feature importances
        feature_importance_df = pd.DataFrame({  # Create a DataFrame
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        print("Top Influential Features:")  # Print top features
        print(feature_importance_df.head(top_n))

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'].head(top_n), feature_importance_df['Importance'].head(top_n))  # Plot top features
        plt.xlabel("Feature Importance")  # X-axis label
        plt.ylabel("Feature")  # Y-axis label
        plt.title(f"Top {top_n} Influential Features")  # Plot title
        plt.gca().invert_yaxis()  # Invert y-axis
        plt.tight_layout()
        plt.show()
    else:
        print("The provided model does not support feature importance.")  # Print unsupported message


def main():
    df = pd.read_csv('gemel_net_dataset.csv')  # Load dataset
    df_preprocessed, label_encoders = preprocessing(df)  # Preprocess data
    X = df_preprocessed.drop(columns=['FUND_CLASSIFICATION'])  # Features
    y = df_preprocessed['FUND_CLASSIFICATION']  # Target

    imputer = SimpleImputer(strategy="median")  # Handle missing values
    X_imputed = imputer.fit_transform(X)  # Impute missing values

    scaler = StandardScaler()  # Scale features
    X_scaled = scaler.fit_transform(X_imputed)  # Transform features
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)  # Split data

    param_grids = {  # Define hyperparameter grids
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        },
        "XGBoost": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    }
    base_models = {  # Define base models
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss'),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(random_state=42, class_weight='balanced', probability=True)
    }
    results = {}  # Initialize results dictionary
    best_xgboost_model = None
    for model_name, model in base_models.items():
        if model_name in param_grids:  # Apply GridSearch for models with parameter grids
            print(f"Performing GridSearch for {model_name}...")
            grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)  # Fit GridSearch
            best_model = grid_search.best_estimator_  # Get best model
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        else:
            best_model = model  # Use default model
            best_model.fit(X_train, y_train)  # Train model

        metrics = train_and_evaluate_model(best_model, X_train, y_train, X_test, y_test)  # Evaluate model
        results[model_name] = metrics  # Store results
        print(f"{model_name} Accuracy: {metrics['accuracy']:.4f}")
        print(f"{model_name} F1-Score: {metrics['f1_score']:.4f}")
        print(f"{model_name} Cross-Validation Mean Accuracy: {metrics['cross_val_mean']:.4f}")
        print(f"{model_name} Classification Report:\n{metrics['classification_report']}\n")

        if model_name == "XGBoost":
            best_xgboost_model = best_model  # Store best XGBoost model

    plot_comparison(results)  # Compare results

    if best_xgboost_model:  # Display feature importance for XGBoost
        display_feature_importance(best_xgboost_model, X.columns)


if __name__ == '__main__':
    main()
