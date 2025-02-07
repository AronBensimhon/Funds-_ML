import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics.pairwise import cosine_similarity


def preprocessing(df):
    """
    Contains data cleaning (missing values, irrelevant features),
    feature engineering and encoding to prep the data for the model.
    """
    drop_features = [  # Columns with excessive missing values or unnecessary for classification
        'AVG_DEPOSIT_FEE', 'ALPHA', 'YIELD_TRAILING_5_YRS',
        'AVG_ANNUAL_YIELD_TRAILING_5YRS', 'STANDARD_DEVIATION',
        'SHARPE_RATIO', 'YIELD_TRAILING_3_YRS', 'AVG_ANNUAL_YIELD_TRAILING_3YRS',
        'MANAGING_CORPORATION_LEGAL_ID', 'CONTROLLING_CORPORATION', 'FUND_ID']
    df = df.drop(columns=drop_features)
    df = df.dropna(subset=['MONTHLY_YIELD', 'YEAR_TO_DATE_YIELD'],
                   how='all')
    num_features = ['DEPOSITS', 'WITHDRAWLS', 'INTERNAL_TRANSFERS', 'NET_MONTHLY_DEPOSITS',
                    'TOTAL_ASSETS', 'AVG_ANNUAL_MANAGEMENT_FEE', 'MONTHLY_YIELD',
                    'YEAR_TO_DATE_YIELD', 'LIQUID_ASSETS_PERCENT', 'STOCK_MARKET_EXPOSURE',
                    'FOREIGN_EXPOSURE', 'FOREIGN_CURRENCY_EXPOSURE']
    for feature in num_features:
        df[feature] = df.groupby('FUND_NAME')[feature].transform(lambda x: x.fillna(x.mean()))
        # ChatGPT : Identify rows where the feature is still NaN after the group mean fill
        invalid_groups = df.groupby('FUND_NAME')[feature].transform('mean').isnull()
        df = df[~invalid_groups]

    df = df.drop(columns='FUND_NAME')  # creates bias for the model since it contains info present in other features
    df['INCEPTION_DATE'] = pd.to_datetime(df['INCEPTION_DATE'], errors='coerce')
    df['REPORT_PERIOD'] = pd.to_datetime(df['REPORT_PERIOD'].astype(str), format='%Y%m')
    df['TIME_ELAPSED_YEARS'] = ((df['REPORT_PERIOD'] - df['INCEPTION_DATE']).dt.days.fillna(
        0)) / 365.25
    df['TIME_ELAPSED_YEARS'] = df['TIME_ELAPSED_YEARS'].fillna(0)
    df = df.drop(columns=['INCEPTION_DATE', 'REPORT_PERIOD'])

    # # # FEATURE ENGINEERING # # #
    fee_min = df['AVG_ANNUAL_MANAGEMENT_FEE'].min() - 0.01
    fee_max = df['AVG_ANNUAL_MANAGEMENT_FEE'].max() + 0.01

    df['FEE_BUCKET'] = pd.cut(
        df['AVG_ANNUAL_MANAGEMENT_FEE'],
        bins=[fee_min, 0.7, 1.4, fee_max],
        labels=['Low Fee', 'Medium Fee', 'High Fee']
    )
    # ChatGPT: bin splitting
    df['TOTAL_FEES'] = df['AVG_ANNUAL_MANAGEMENT_FEE'] * df['TOTAL_ASSETS']  # Calculate total fees collected

    # Encoding
    categorical_features = ['FUND_CLASSIFICATION', 'TARGET_POPULATION', 'SPECIALIZATION', 'SUB_SPECIALIZATION',
                            'MANAGING_CORPORATION', 'FEE_BUCKET']  # Categorical features
    encoders = {col: LabelEncoder() for col in categorical_features}  # ChatGPT: Create LabelEncoder objects
    for col, encoder in encoders.items():
        df[col] = encoder.fit_transform(df[col])  # Encode categorical features
    return df, encoders


def identify_class_distribution(df):
    class_counts = df['FUND_CLASSIFICATION'].value_counts()
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Class Distribution in FUND_CLASSIFICATION', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Number of Instances', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('graph_results/class_distribution.png')
    # plt.show()
    return class_counts


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Used to run several models with same script
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)  # ChatGPT: print report of model

    cross_val_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')  # Cross-validation
    cv_mean = cross_val_scores.mean()  # Mean cross-validation score
    return {
        "model": model,
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": report,
        "cross_val_mean": cv_mean
    }


def plot_comparison(results):
    """
    Plot of each model performance
    """
    model_names = list(results.keys())
    accuracies = [metrics['accuracy'] for metrics in results.values()]
    f1_scores = [metrics['f1_score'] for metrics in results.values()]
    # ChatGPT: plot config
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, alpha=0.7, label='Accuracy', width=0.4, align='center')  # Plot accuracies
    plt.bar(model_names, f1_scores, alpha=0.7, label='F1-Score', width=0.4, align='edge')  # Plot F1 scores
    plt.xlabel('Model')  # X-axis label
    plt.ylabel('Score')  # Y-axis label
    plt.title('Model Performance Comparison')  # Plot title
    plt.legend()  # Add legend
    plt.tight_layout()
    # plt.show() # Uncomment if you want to allow plot display, needs manual closing for the code to continue running.
    plt.savefig('graph_results/models_comparison.png')
    plt.close()


def display_feature_importance(model, feature_names, top_n=10):
    """
    Plot the features with the highest influence
    """
    if hasattr(model, "feature_importances_"):  # ChatGPT: check if feature_importance method is supported
        feature_importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        print("Top Influential Features:")  # Print top features
        print(feature_importance_df.head(top_n))
        # ChatGPT: plot config
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'].head(top_n),
                 feature_importance_df['Importance'].head(top_n))  # Plot top features
        plt.xlabel("Feature Importance")  # X-axis label
        plt.ylabel("Feature")  # Y-axis label
        plt.title(f"Top {top_n} Influential Features")  # Plot title
        plt.gca().invert_yaxis()  # Invert y-axis
        plt.tight_layout()
        # plt.show()
        plt.savefig('graph_results/feature_importance.png')
        plt.close()
    else:
        print("The provided model does not support feature importance.")


def run_classification(X, X_scaled, y):
    """
    1. GridSearch for parameter tuning
    2. Finding the best model based on accuracy
    3. Display top influencing features
    """
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

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
    base_models = {  # default or basic parameters used, GridSearch will update with optimal params
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss'),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(random_state=42, class_weight='balanced', probability=True)
    }
    results = {}
    best_xgboost_model = None
    for model_name, model in base_models.items():
        if model_name in param_grids:
            print(f"Performing GridSearch for {model_name}...")
            grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_  # implement best params
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        metrics = train_and_evaluate_model(best_model, X_train, y_train, X_test, y_test)
        results[model_name] = metrics
        # ChatGPT prints
        print(f"{model_name} Accuracy: {metrics['accuracy']:.4f}")
        print(f"{model_name} F1-Score: {metrics['f1_score']:.4f}")
        print(f"{model_name} Cross-Validation Mean Accuracy: {metrics['cross_val_mean']:.4f}")
        print(f"{model_name} Classification Report:\n{metrics['classification_report']}\n")
        if model_name == "XGBoost":
            best_xgboost_model = best_model
    plot_comparison(results)  # Compare results
    if best_xgboost_model:
        display_feature_importance(best_xgboost_model, X.columns)


def run_clustering(X):
    """
    1. Testing Kmeans, Hierarchical and DBSCAN
    2. Finding the best clustering model based on accuracy
    3. Compare results
    """
    # KMeans
    print("\nPerforming Kmeans Clustering")
    wcss = []  # To store Within-Cluster Sum of Squares
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # ChatGPT: plot config
    # Plot the Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method: Optimal Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.grid()
    # plt.show()
    plt.savefig('graph_results/elbow_method_res.png')
    plt.close()

    optimal_k = 3  # based on visual inspection of the elbow plot
    print(f"Optimal number of clusters after elbow method evaluation : {optimal_k}")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)

    # Hierarchical
    print("\nPerforming Hierarchical Clustering")
    linkage_matrix = linkage(X, method='ward')  # Ward's method minimizes variance within clusters

    # ChatGPT: plot config
    # Plot the dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Points or Clusters")
    plt.ylabel("Distance")
    # plt.show()
    plt.savefig('graph_results/hierarchical_dendrogram.png')
    plt.close()

    optimal_threshold = 100  # based on dendrogram inspection
    hierarchical_labels = fcluster(linkage_matrix, t=optimal_threshold, criterion='distance')
    num_clusters_hierarchical = len(np.unique(hierarchical_labels))
    print(f"Number of clusters found using hierarchical clustering"
          f" (height={optimal_threshold}): {num_clusters_hierarchical}")

    # DBSCAN
    print("\nPerforming Hierarchical Clustering")
    print(f"Optimal epsilon after test iterations : 3")
    eps = 3  # This value was initially set to 7 then reduced till DBSCAN successfully detected 3 clusters
    min_samples = 10  # Minimum number of samples
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X)
    num_clusters_dbscan = len(np.unique(dbscan_labels[dbscan_labels != -1]))  # Exclude noise
    num_noise_points = np.sum(dbscan_labels == -1)
    print(f"DBSCAN: {num_clusters_dbscan} clusters found, {num_noise_points} noise points")

    # ChatGPT: Plot the DBSCAN results
    plt.figure(figsize=(10, 6))
    unique_labels = set(dbscan_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points
            color = [0, 0, 0, 1]  # Black for noise
        mask = (dbscan_labels == label)
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], label=f'Cluster {label}' if label != -1 else 'Noise')
    plt.title("DBSCAN Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('graph_results/dbscan_clusters.png')
    plt.close()

    # Analyze cluster distribution
    kmeans_cluster_counts = pd.Series(kmeans_labels).value_counts()
    hierarchical_cluster_counts = pd.Series(hierarchical_labels).value_counts()
    dbscan_cluster_counts = pd.Series(dbscan_labels[dbscan_labels != -1]).value_counts()

    print("\nCluster Distribution:")
    print(f"KMeans Cluster Distribution:\n{kmeans_cluster_counts}")
    print(f"DBSCAN Cluster Distribution (excluding noise):\n{dbscan_cluster_counts}")
    print(f"Hierarchical Cluster Distribution:\n{hierarchical_cluster_counts}")

    print("\nComparing Clustering Results")
    methods = ['KMeans', 'DBSCAN', 'Hierarchical']
    num_clusters = [len(kmeans_cluster_counts), num_clusters_dbscan, num_clusters_hierarchical]

    # ChatGPT: plot config
    plt.figure(figsize=(8, 5))
    plt.bar(methods, num_clusters, color=['blue', 'orange', 'green'])
    plt.title('Comparison of Clustering Methods')
    plt.ylabel('Number of Clusters')
    plt.xlabel('Clustering Method')
    # plt.show()
    plt.savefig('graph_results/clusters_comparison.png')
    plt.close()
    return {
        "KMeans": {
            "labels": kmeans_labels,
            "num_clusters": len(kmeans_cluster_counts),
            "cluster_distribution": kmeans_cluster_counts
        },
        "Hierarchical": {
            "labels": hierarchical_labels,
            "num_clusters": num_clusters_hierarchical,
            "cluster_distribution": hierarchical_cluster_counts
        },
        "DBSCAN": {
            "labels": dbscan_labels,
            "num_clusters": num_clusters_dbscan,
            "num_noise_points": num_noise_points,
            "cluster_distribution": dbscan_cluster_counts
        }
    }


def run_anomaly_detection(X, df):
    """
    Perform anomaly detection using Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM.
    Results include detection of anomalies, model comparison, and identification of anomalies in the raw dataset.
    """
    print("\nPerforming Anomaly Detection")
    # Isolation Forest
    isolation_forest = IsolationForest(
        contamination=0.01, max_samples=0.5, n_estimators=50, random_state=42)
    isolation_labels = isolation_forest.fit_predict(X)
    isolation_anomalies = sum(isolation_labels == -1)  # ChatGPT: sum anomalies
    print(f"Isolation Forest detected {isolation_anomalies} anomalies.")

    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.01)
    lof_labels = lof.fit_predict(X)
    lof_anomalies = sum(lof_labels == -1)
    print(f"Local Outlier Factor detected {lof_anomalies} anomalies.")

    # One-Class SVM
    ocsvm = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
    ocsvm_labels = ocsvm.fit_predict(X)
    ocsvm_anomalies = sum(ocsvm_labels == -1)
    print(f"One-Class SVM detected {ocsvm_anomalies} anomalies.")

    print("\nComparing Results Across Models")
    anomaly_methods = ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM']
    anomaly_counts = [isolation_anomalies, lof_anomalies, ocsvm_anomalies]

    # ChatGPT: plot config
    plt.figure(figsize=(8, 5))
    plt.bar(anomaly_methods, anomaly_counts, color=['green', 'red', 'blue'])
    plt.title('Number of Anomalies Detected by Each Model')
    plt.ylabel('Number of Anomalies')
    plt.xlabel('Anomaly Detection Method')
    # plt.show()
    plt.savefig('graph_results/anomalies_res.png')
    plt.close()

    print("\nIdentifying Anomalies Using the Best Model")
    best_model = 'Isolation Forest'
    best_labels = isolation_labels
    anomalies = df[best_labels == -1]

    print(f"Best Model for anomaly detection: {best_model}")
    print("\nDetermining Features with Most Anomalies")
    feature_anomaly_counts = anomalies.apply(lambda col: col.value_counts().get(-1, 0), axis=0)
    top_features = feature_anomaly_counts.sort_values(ascending=False).head(5).index.tolist()
    print("Features with Most Anomalies:")
    print(top_features)
    return {
        "Isolation Forest": {
            "anomalies": df[isolation_labels == -1],
            "num_anomalies": isolation_anomalies
        },
        "Local Outlier Factor": {
            "anomalies": df[lof_labels == -1],
            "num_anomalies": lof_anomalies
        },
        "One-Class SVM": {
            "anomalies": df[ocsvm_labels == -1],
            "num_anomalies": ocsvm_anomalies
        }
    }


def recommend_similar_funds(df, fund_name, top_n=3):
    """
    Recommends the top N most similar funds using content-based filtering and cosine similarity. This function
    targets performance-risk metrics such as alpha, sharpe ratio, and standard deviation to provide recommendations
    based on a performance-risk scale.

    Read more about Sharpe Ratio and Alpha here:
    https://www.investopedia.com/ask/answers/021315/what-difference-between-sharpe-ratio-and-alpha.asp
    """
    num_features = ['ALPHA', 'STANDARD_DEVIATION', 'SHARPE_RATIO']
    cat_feature = ['FUND_NAME']
    df = df.dropna(subset=num_features, how='all').copy()
    for feature in num_features:
        df[feature] = df.groupby(cat_feature)[feature].transform(
            lambda x: x.fillna(x.mean())
        )
    df[cat_feature] = df[cat_feature].fillna('Unknown')
    df = df.drop_duplicates(subset='FUND_NAME', keep='first')
    relevant_columns = num_features + cat_feature
    df = df[relevant_columns].copy()
    scaler = StandardScaler()
    numerical_data = scaler.fit_transform(df[num_features])
    categorical_data = pd.get_dummies(df[cat_feature])
    combined_features = np.hstack([numerical_data, categorical_data.values])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(combined_features)
    similarity_df = pd.DataFrame(
        cosine_sim, index=df['FUND_NAME'], columns=df['FUND_NAME']
    )
    if fund_name not in similarity_df.index:
        raise ValueError(f"Fund name '{fund_name}' not found in the dataset.")

    similarity_series = similarity_df.loc[fund_name]
    similar_funds = (
        similarity_series
        .drop(fund_name, errors='ignore')  # Exclude the fund itself
        .pipe(pd.Series.sort_values, ascending=False)  # Explicitly use pandas.Series.sort_values
        .head(top_n)
    )
    return list(similar_funds.index)


def main():
    df = pd.read_csv('gemel_net_dataset.csv')

    # # # PREPROCESSING # # #
    print('\n===== Running preprocessing =====')
    df_preprocessed, label_encoders = preprocessing(df)
    distribution = identify_class_distribution(df_preprocessed)
    print('\nClass distribution after preprocessing : ', distribution)
    X = df_preprocessed.drop(columns=['FUND_CLASSIFICATION'])
    y = df_preprocessed['FUND_CLASSIFICATION']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # # # # CLASSIFICATION # # #
    print('\n===== Running classification model =====')
    run_classification(X, X_scaled, y)

    # # # # #  UNSUPERVISED ANALYSIS  # # #
    print('\n===== Running unsupervised analysis =====')
    print("\nApplying PCA for Dimensionality Reduction")
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)
    run_clustering(X_pca)
    run_anomaly_detection(X_pca, df_preprocessed)

    # # # # RECOMMENDER SYSTEM # # #
    print('\n===== Running recommender system =====')
    funds = df['FUND_NAME'].unique().tolist()
    fund_name = 'סלייס קופת גמל להשקעה אג"ח ללא מניות'
    # fund name example, print(random.choice(funds)) to try another fund.
    similar_funds = recommend_similar_funds(df, fund_name, top_n=3)
    print(f"Top {len(similar_funds)} funds similar to '{fund_name}' are:")
    for rec in similar_funds:
        print(rec)


if __name__ == '__main__':
    main()

