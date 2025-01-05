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
    # Drop columns with excessive missing values or unnecessary for classification
    drop_features = [
        'AVG_DEPOSIT_FEE', 'ALPHA', 'YIELD_TRAILING_5_YRS',
        'AVG_ANNUAL_YIELD_TRAILING_5YRS', 'STANDARD_DEVIATION',
        'SHARPE_RATIO', 'YIELD_TRAILING_3_YRS', 'AVG_ANNUAL_YIELD_TRAILING_3YRS',
        'MANAGING_CORPORATION_LEGAL_ID', 'CONTROLLING_CORPORATION', 'FUND_NAME', 'FUND_ID'
    ]
    df = df.drop(columns=drop_features)

    # Drop rows where essential features are missing
    df = df.dropna(subset=['MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD'], how='all')

    # Fill missing values in numerical features with median
    num_features = ['AVG_ANNUAL_MANAGEMENT_FEE', 'MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD']
    df[num_features] = df[num_features].fillna(df[num_features].median())

    # Feature Engineering: Time Elapsed in Years
    df['INCEPTION_DATE'] = pd.to_datetime(df['INCEPTION_DATE'], errors='coerce')
    df['REPORT_PERIOD'] = pd.to_datetime(df['REPORT_PERIOD'].astype(str), format='%Y%m')
    df['TIME_ELAPSED_YEARS'] = ((df['REPORT_PERIOD'] - df['INCEPTION_DATE']).dt.days.fillna(0)) / 365.25
    df['TIME_ELAPSED_YEARS'] = df['TIME_ELAPSED_YEARS'].fillna(0)

    # Drop original date columns after feature creation
    df = df.drop(columns=['INCEPTION_DATE', 'REPORT_PERIOD'])

    # Feature Engineering: Bucketization for AVG_ANNUAL_MANAGEMENT_FEE
    df['FEE_BUCKET'] = pd.cut(
        df['AVG_ANNUAL_MANAGEMENT_FEE'],
        bins=[-1, 0.5, 1.5, float('inf')],
        labels=['Low Fee', 'Medium Fee', 'High Fee']
    )

    # Feature Engineering: Total Fees Collected
    df['TOTAL_FEES'] = df['AVG_ANNUAL_MANAGEMENT_FEE'] * df['TOTAL_ASSETS']

    # Perform encoding after all feature engineering
    categorical_features = ['FUND_CLASSIFICATION', 'TARGET_POPULATION', 'SPECIALIZATION', 'SUB_SPECIALIZATION',
                            'MANAGING_CORPORATION', 'FEE_BUCKET']

    # ChatGPT: Create a dictionary of LabelEncoder objects for each categorical feature
    encoders = {col: LabelEncoder() for col in categorical_features}

    # Encode each categorical column with its corresponding LabelEncoder
    for col, encoder in encoders.items():
        df[col] = encoder.fit_transform(df[col])  # Map unique categories to integers and replace the original column

    return df, encoders


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)

    cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cross_val_scores.mean()
    return {
        "model": model,
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": report,
        "cross_val_mean": cv_mean
    }


def plot_comparison(results):
    model_names = list(results.keys())
    accuracies = [metrics['accuracy'] for metrics in results.values()]
    f1_scores = [metrics['f1_score'] for metrics in results.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, alpha=0.7, label='Accuracy', width=0.4, align='center')
    plt.bar(model_names, f1_scores, alpha=0.7, label='F1-Score', width=0.4, align='edge')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()


def display_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, "feature_importances_"):  # Check if the model has feature_importances_ method
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Display the top features
        print("Top Influential Features:")
        print(feature_importance_df.head(top_n))

        # Plot the top features
        plt.figure(figsize=(10, 6))
        plt.barh(
            feature_importance_df['Feature'].head(top_n),
            feature_importance_df['Importance'].head(top_n)
        )
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Influential Features")
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization
        plt.tight_layout()
        plt.show()
    else:
        print("The provided model does not support feature importance.")


def main():
    df = pd.read_csv('gemel_net_dataset.csv')
    df_preprocessed, label_encoders = preprocessing(df)
    X = df_preprocessed.drop(columns=['FUND_CLASSIFICATION'])
    y = df_preprocessed['FUND_CLASSIFICATION']

    imputer = SimpleImputer(strategy="median")  # Handling missing values
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()  # Scaling features
    X_scaled = scaler.fit_transform(X_imputed)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # ChatGPT: Define hyperparameter grids
    param_grids = {
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
    base_models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss'),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000),  # Increased max_iter
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(random_state=42, class_weight='balanced', probability=True)
    }
    # Train and evaluate each model with GridSearchCV where applicable
    results = {}
    best_xgboost_model = None
    for model_name, model in base_models.items():
        if model_name in param_grids:  # Apply GridSearch only to models with parameter grids
            print(f"Performing GridSearch for {model_name}...")
            grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        metrics = train_and_evaluate_model(best_model, X_train, y_train, X_test, y_test)
        results[model_name] = metrics
        print(f"{model_name} Accuracy: {metrics['accuracy']:.4f}")
        print(f"{model_name} F1-Score: {metrics['f1_score']:.4f}")
        print(f"{model_name} Cross-Validation Mean Accuracy: {metrics['cross_val_mean']:.4f}")
        print(f"{model_name} Classification Report:\n{metrics['classification_report']}\n")

        if model_name == "XGBoost":
            best_xgboost_model = best_model

    # Compare results
    plot_comparison(results)

    # Display top influential features for XGBoost
    if best_xgboost_model:
        display_feature_importance(best_xgboost_model, X.columns)


if __name__ == '__main__':
    main()
