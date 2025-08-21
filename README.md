

# Breast Cancer Prediction

This project is a **machine learning pipeline** for predicting breast cancer using patient tumor data. It combines **data analysis, feature selection, model training**, and a **user-friendly front-end interface**.

## Overview

The main steps of this project include:

1. **Exploratory Data Analysis (EDA)**

   * Analyzed the dataset to understand distributions, correlations, and important features.
   * Visualized key relationships between features and the target variable.

2. **Model Training**

   * Trained **Random Forest** and **XGBoost** classifiers on the dataset.
   * Evaluated models using metrics like **accuracy, precision, recall, and F1-score**.
   * Performed **feature selection** to identify the most important features.

3. **User-Friendly Interface**

   * Instead of requiring users to input all **17 features**, the interface allows **only 3 key inputs**:

     * `concave points_worst`
     * `perimeter_worst`
     * `radius_worst`
   * The remaining 14 features are predicted automatically using **Random Forest regression models** (feature imputation).
   * The complete 17-feature vector is then used by the **main Random Forest classifier** with the accuracy of 96% to predict whether the patient has cancer.

4. **Prediction**

   * The final output indicates **Benign** or **Malignant** along with a **confidence score**.

## Key Features Used

* radius\_mean, texture\_mean, perimeter\_mean, area\_mean, compactness\_mean
* concavity\_mean, concave points\_mean, radius\_se, area\_se
* radius\_worst, texture\_worst, perimeter\_worst, area\_worst
* smoothness\_worst, compactness\_worst, concavity\_worst, concave points\_worst

*(Only 3 user inputs are required; the rest are automatically estimated.)*

## Technologies Used

* Python 3.x
* Pandas, NumPy
* Scikit-learn (Random Forest, XGBoost)
* Streamlit for interactive front-end
* Pickle for model serialization

4. Input the **3 key features** and get the **prediction**.

## Notes

* The project demonstrates **feature imputation via regression models** to simplify user input.
* The pipeline achieves high accuracy while keeping the interface simple and intuitive.

