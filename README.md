# PCOS Prediction Using Machine Learning

# Project Overview
This project uses machine learning models to predict whether an individual has Polycystic Ovary Syndrome (PCOS) based on various features such as BMI, cycle length, hormone levels, and other relevant factors. The model can help doctors and individuals monitor potential health conditions related to PCOS.

# Features
BMI: Body Mass Index of the individual
Cycle Length (days): Length of the menstrual cycle in days
LH (mIU/mL): Luteinizing hormone levels
FSH/LH: Ratio of Follicle Stimulating Hormone to Luteinizing Hormone
AMH (ng/mL): Anti-Müllerian Hormone levels
Weight Gain (Y/N): Whether the individual has gained weight (Y=1, N=0)
Hair Growth (Y/N): Whether the individual experiences abnormal hair growth (Y=1, N=0)
Follicle No. (L): Number of follicles in the left ovary
Follicle No. (R): Number of follicles in the right ovary
The target variable is PCOS (Y/N), where Y indicates the presence of PCOS and N indicates its absence.

# Key Insights
Exploratory Data Analysis (EDA) was performed to understand the data distribution, correlations, and the potential impact of various features on PCOS.

Data Preprocessing included handling missing values, encoding categorical variables (like 'Y/N' features), and scaling the features for models that require normalized data (like Logistic Regression and SVM).

Multiple Classification Models were applied, including:
Logistic Regression
Support Vector Machine (SVM)
Naive Bayes (GaussianNB)
Random Forest
XGBoost

# Results
Accuracy scores were evaluated for each model to determine which one performed best for the PCOS prediction.
The Naive Bayes model was found to give the best results based on the dataset provided.

# Gradio Interface
A Gradio interface was created to allow users to input their data (e.g., BMI, cycle length, hormone levels) and receive a prediction on whether they are at risk for PCOS.

# Input Features
BMI
Cycle length (days)
LH (mIU/mL)
FSH/LH
AMH (ng/mL)
Weight Gain (Y/N)
Hair Growth (Y/N)
Follicle No. (L)
Follicle No. (R)

# Output
PCOS Positive or PCOS Negative
The Gradio interface provides a user-friendly method to interact with the model and make predictions based on input data.

# Setup and Installation
Requirements
Python 3.7 or higher
Required Python libraries:
pandas
numpy
scikit-learn
matplotlib
seaborn
gradio
xgboost (optional, for XGBoost model)
statsmodels

Clone the repository and install the dependencies using the following commands:

git clone <repository-url>
cd <project-directory>
pip install -r requirements.txt

# Running the Project
To run the notebook for model training and prediction:

Open the notebook:
Open the .ipynb file using a Jupyter notebook environment.

To run the Gradio interface (for prediction):

The interface will be launched as a web app where you can input values for the features (BMI, cycle length, hormone levels, etc.) and get predictions.

# Load the model
model = joblib.load('pcos_model.pkl')

# Making predictions with the loaded model
prediction = model.predict(input_data)

