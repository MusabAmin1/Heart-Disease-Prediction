Heart Disease Prediction

1- Overview:

This project predicts the likelihood of heart disease in a patient using Machine Learning. A Stacked Logistic Regression model (ensemble of XGBoost, SVC, and Logistic Regression) is trained on a Kaggle heart dataset. Users can input patient details through a Streamlit app, and the model outputs the prediction.

2- Files:

heart.csv – Original heart disease dataset
Heart_EDA_Modeling.ipynb – Notebook with data cleaning, EDA, feature engineering, and model training
Heart_UI.py – Streamlit app for interactive predictions
heart_stacked_lr_model.pkl – Trained stacked model
heart_scale.pkl – StandardScaler used for scaling features
README.md – Project description

3- Installation:

Install Python (recommended ≥ 3.10)
Install required libraries:
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib

4- Usage:

Clone the repository:
git clone <repository_url>
cd <project_folder>
Run the Streamlit app:
streamlit run Heart_UI.py
Input patient details and get the heart disease prediction.

5- Model Information:

Algorithm: Stacked Logistic Regression (XGBoost + SVC + Logistic Regression)
Final Accuracy: ~87.5% on test data
Confusion Matrix:
[[ 61  11]
 [ 12 100]]
Key Features Used:
Age, RestingBP, Cholesterol, MaxHR, Oldpeak
ST_Slope_Up, ST_Slope_Flat, isExerciseAngina
ChestPainType_ATA, ChestPainType_NAP, isMale, isFastingBS
RestingECG_ST, RestingECG_Normal

6- Notes:

Data cleaning included fixing invalid RestingBP, Cholesterol, and negative Oldpeak values.
Categorical variables were converted to numerical dummy variables.
Model selection included Base models, Tuned models, Bagging, XGBoost, and final stacked ensemble.
