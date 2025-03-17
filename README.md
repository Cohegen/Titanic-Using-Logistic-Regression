# Titanic Survival Prediction using Logistic Regression

## Overview
This project applies Logistic Regression to predict passenger survival on the Titanic based on features such as age, fare, class, and other characteristics. The dataset used is the famous Titanic dataset.

## Dataset
The dataset contains information about passengers, including:
- Passenger ID
- Age
- Sex
- Ticket class
- Fare
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Embarked location

The target variable is **Survived**, where:
- `0` = Passenger did not survive
- `1` = Passenger survived

## Project Steps
### 1. Importing Libraries
The necessary libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn are imported.

### 2. Loading the Dataset
The dataset is loaded using Pandas and displayed for initial inspection.

### 3. Exploratory Data Analysis (EDA)
- Checking for missing values
- Visualizing distributions of age, fare, and class
- Comparing survival rates based on different features

### 4. Data Preprocessing
- Handling missing values in Age, Embarked, and Fare columns
- Encoding categorical variables (Sex, Embarked)
- Scaling numerical features for better model performance

### 5. Model Training (Logistic Regression)
- Splitting the dataset into training and testing sets
- Training a Logistic Regression model using Scikit-Learn
- Evaluating the model’s accuracy and performance

### 6. Hyperparameter Tuning
Using GridSearchCV or RandomizedSearchCV to find the best hyperparameters for Logistic Regression.

## Results
- Model accuracy on the test set
- Confusion matrix for performance evaluation
- Precision, Recall, and F1-score analysis

## Requirements
To run this project, install the required libraries using:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Usage
Run the Jupyter Notebook (`titanic_log_reg.ipynb`) step by step to train and evaluate the model.

## Future Improvements
- Experimenting with other machine learning models (Random Forest, SVM, XGBoost)
- Feature engineering for better accuracy
- Handling class imbalance using oversampling or undersampling techniques

