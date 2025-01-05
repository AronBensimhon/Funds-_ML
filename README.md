# Funds_ML

Welcome to the **Funds_ML** repository! This project demonstrates an end-to-end classification pipeline including an unsupervised analysis of financial funds data. The primary focus is on preprocessing, feature engineering, and model evaluation, leveraging advanced machine learning models and visualization techniques. In a second part you will find a clustering and anomaly detection analysis. 

## Features

- **Preprocessing Pipeline**: Handles missing values, irrelevant features, and encodes categorical variables.
- **Feature Engineering**: Creates new features like time elapsed, fee buckets, and total fees collected.
- **Model Training and Evaluation**: Includes several classifiers such as Random Forest, XGBoost, Logistic Regression, K-Nearest Neighbors, and Support Vector Machines.
- **Hyperparameter Optimization**: Uses GridSearchCV to find the best parameters for Random Forest and XGBoost models.
- **Visualization**:
  - Comparison of model performance.
  - Display of influential features for models supporting feature importance.

## Repository Structure

- `main.py`: Main script containing the pipeline implementation.
- `gemel_net_dataset.csv`: Dataset used for training and testing (not included in the repository; see below for details on dataset usage).

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

## How to Use

1. Run the script:

   ```bash
   python main.py
   ```

2. The script will preprocess the dataset, train multiple machine learning models, evaluate them, and visualize the results.

3. Output includes:
   - Accuracy, F1-Score, and cross-validation accuracy for each model.
   - A bar chart comparing model performance.
   - A list and visualization of the top influential features for the XGBoost model.

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

## Dataset

The dataset used in this project is from the **GemelNet** project. It includes features related to financial fund performance, such as management fees, yields, and asset sizes. For privacy and licensing reasons, the dataset is not provided in this repository. Please refer to the official GemelNet dataset for more details.

## Future Enhancements

- Add support for additional machine learning models.
- Implement automated exploratory data analysis (EDA).
- Extend feature engineering to capture more financial insights.
- Integrate deep learning models for enhanced performance.

## Author

This repository is maintained by **Aron Bensimhon**. If you have any questions or feedback, feel free to contact me through GitHub.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Special thanks to the GemelNet project for providing the dataset used in this analysis.

