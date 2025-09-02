# Loan Approval Prediction Project

## Objective
Build a machine learning model to predict whether a loan application will be approved based on applicant information and financial data.

## Dataset
- **Source**: Loan Approval Dataset (Kaggle or similar)  
- **Problem Type**: Binary Classification  
- **Focus**: Handling imbalanced data and evaluating performance with precision, recall, and F1-score  

## Tools & Libraries
- **Python**  
- **Pandas**: Data manipulation  
- **Scikit-learn**: Preprocessing, modeling, evaluation  
- **Imbalanced-learn (SMOTE)**: Balancing classes  
- **Matplotlib**: Visualization  

---

## Steps

### 1. Data Loading
- Load dataset with `pandas.read_csv()`  
- Check missing values with `df.isnull().sum()`  
- Inspect dataset columns and target distribution  

### 2. Feature Engineering
Created new features that capture financial relationships:
- **Education Encoding**: Graduate = 1, Not Graduate = 0  
- **Self-Employed Encoding**: Yes = 1, No = 0  
- **Loan-to-Income Ratio**: `loan_amount / income_annum`  
- **Total Assets**: Sum of residential, commercial, luxury, and bank asset values  
- **Assets-to-Loan Ratio**: `total_assets / loan_amount`  

### 3. Train-Test Split
- Split into train (80%) and test (20%)  
- Stratified split to preserve class balance  

### 4. Handle Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)**  
- Balanced the training set for better model learning  

### 5. Feature Scaling
- Standardized numerical features with `StandardScaler`  

### 6. Model Training
- **Logistic Regression**  
- **Decision Tree Classifier**  

### 7. Evaluation
- **Metrics Used**: Precision, Recall, F1-score, Confusion Matrix  
- Compared Logistic Regression and Decision Tree on weighted F1-score  

### 8. Feature Importance
- Extracted feature importance from Decision Tree  
- Visualized top 10 most important features  

---

## Results

- Both models trained successfully  
- Evaluation metrics highlight trade-offs between Logistic Regression and Decision Tree  
- Feature importance plot reveals which factors influence loan approval the most  

---
