# Machine Learning Project🚀

This repository contains a collection of Python scripts implementing fundamental machine learning, data processing, and data science techniques. These scripts were developed as part of a Machine Learning lab curriculum.

## 📂 Repository Contents

Here is a breakdown of the lab exercises included in this repository:

###Data Cleaning and Handling Missing Values (`lab1.py` / `main.py`)
*   **Objective:** Clean a semi-structured dataset containing inconsistencies, missing values, and mixed formats.
*   **Techniques Used:** 
    * Handling missing values (Mean/Median/Mode imputation).
    * Outlier detection and capping using the Interquartile Range (IQR) method.
    * Standardizing text and categorical data (Regex).
    * Data visualization (Missing values bar chart, Boxplots).

###Feature Engineering & Preprocessing (`lab2.py`)
*   **Objective:** Prepare raw data for machine learning algorithms.
*   **Techniques Used:**
    * **Feature Engineering:** Date transformation, Discretization (Binning), One-Hot Encoding.
    * **Feature Selection:** Correlation Analysis, `SelectKBest` (Statistical selection).
    * **Feature Scaling:** `StandardScaler`, `MinMaxScaler`.

###Implementing & Evaluating Regression Models (`lab3_regression.py`)
*   **Objective:** Predict continuous values using a synthetic house pricing dataset.
*   **Techniques Used:**
    * Simple Linear Regression.
    * Multiple Linear Regression.
    * Model Evaluation Metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).

###Implementing Classification Models (`lab4_classification.py`)
*   **Objective:** Classify data using the Breast Cancer Wisconsin dataset.
*   **Models Implemented:** 
    * Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN).
*   **Evaluation Metrics:** Accuracy, Precision, Sensitivity (Recall), Specificity, and F1-Score.

#K-Means Clustering (`lab5_kmeans.py`)
*   **Objective:** Group unsupervised data into distinct segments (Customer Segmentation).
*   **Techniques Used:**
    * Determining optimal clusters using the **Elbow Method**.
    * Assigning cluster labels and visualizing cluster centroids.

---

## 🛠️ Requirements & Installation

To run the scripts in this repository, you will need Python installed along with the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
