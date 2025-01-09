# Funds_ML

Welcome to the **Funds_ML** repository! This project demonstrates an end-to-end machine learning pipeline, analyzing financial funds data through classification, clustering, and anomaly detection. The focus is on preprocessing, feature engineering, and evaluating multiple models, with rich visualizations and insights into fund behavior.

---

## Features

### **Data Preprocessing**
- Handles missing values using median imputation.
- Removes irrelevant and redundant features.
- Encodes categorical variables with `LabelEncoder`.
- Creates engineered features such as:
  - **Time elapsed** since fund inception.
  - **Fee buckets** (Low, Medium, High) based on management fees.
  - **Total fees collected**.

### **Classification**
- Trains multiple machine learning models, including:
  - **Random Forest**
  - **XGBoost**
  - **Logistic Regression**
  - **K-Nearest Neighbors**
  - **Support Vector Machine**
- Uses **GridSearchCV** for hyperparameter optimization of Random Forest and XGBoost.
- Evaluates models with:
  - Accuracy
  - F1-Score
  - Cross-validation scores
  - Classification reports
- Visualizes the top influential features for models supporting feature importance.

### **Unsupervised Learning**
#### **Clustering**
- Implements clustering algorithms:
  - **KMeans** (Elbow method for optimal clusters)
  - **Hierarchical Clustering** (Dendrograms for threshold selection)
  - **DBSCAN** (Density-Based Spatial Clustering)
- Compares clustering results using metrics and visualizations.

#### **Anomaly Detection**
- Detects anomalies using:
  - **Isolation Forest**
  - **Local Outlier Factor (LOF)**
  - **One-Class SVM**
- Visualizes anomaly detection results and highlights features with the most anomalies.

---

## Repository Structure

- `classification_pipeline.py`: Main script containing the pipeline implementation.
- `gemel_net_dataset.csv`: Dataset used for training and testing (not included in the repository; see below for details on dataset usage).
- `requirements.txt`: List of dependencies required for the project.
- `feature_meaning.csv` Explanation of each feature appearing in the dataset for better understanding.

---

## Setup Instructions

### Prerequisites

Ensure you have Python 3.7+ installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AronBensimhon/Funds_ML.git
   cd Funds_ML
   ```

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset (`gemel_net_dataset.csv`) in the root directory.

---

## How to Use

1. Run the script:

   ```bash
   python main.py
   ```

2. The script will preprocess the dataset, train multiple machine learning models, evaluate them, and visualize the results.

3. Outputs include:
   - Accuracy, F1-Score, and cross-validation accuracy for each model.
   - A bar chart comparing model performance.
   - A list and visualization of the top influential features for the XGBoost model.
   - Clustering and anomaly detection results.

---

## Key Functions

### Preprocessing

The `preprocessing` function:
- Drops irrelevant or redundant columns.
- Fills missing values using median imputation.
- Performs feature engineering to create meaningful new features.
- Encodes categorical features using `LabelEncoder`.

### Model Training and Evaluation

The `train_and_evaluate_model` function:
- Trains a specified model on the training data.
- Evaluates the model on the test set using accuracy, F1-Score, and a classification report.
- Conducts cross-validation for robust performance metrics.

### Visualization

- **`plot_comparison`**: Compares accuracy and F1-Score across models.
- **`display_feature_importance`**: Displays and plots the most important features for the XGBoost model.

### Clustering

- Uses KMeans, Hierarchical Clustering, and DBSCAN to group data into clusters.
- Visualizes the elbow method, dendrogram, and DBSCAN results for optimal cluster evaluation.

### Anomaly Detection

- Identifies anomalies using Isolation Forest, LOF, and One-Class SVM.
- Highlights the features with the highest number of anomalies.

---

## Dataset

The dataset used in this project is from the **GemelNet** project. It includes features related to financial fund performance, such as management fees, yields, and asset sizes. For privacy and licensing reasons, the dataset is not provided in this repository. Please refer to the official GemelNet dataset for more details.

---

## Future Enhancements

- Add support for additional machine learning models.
- Implement automated exploratory data analysis (EDA).
- Extend feature engineering to capture more financial insights.
- Integrate deep learning models for enhanced performance.

---

## Author

This repository is maintained by **Aron Bensimhon**. If you have any questions or feedback, feel free to contact me through GitHub.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

Special thanks to the GemelNet project for providing the dataset used in this analysis.

