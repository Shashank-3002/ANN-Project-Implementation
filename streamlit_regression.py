import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('churn_regression_model.h5')

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

# Define the Streamlit app
st.title('Customer Churn Regression Prediction')
# Define input fields for the features
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', min_value=18, max_value=100)
balance = st.number_input('Balance')
creditscore = st.number_input('Credit Score')
estimatedsalary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', min_value=0, max_value=10, value=2)
numofproducts = st.slider('Number of Products', min_value=1, max_value=4, value=1)
hascrcard = st.selectbox('Has Credit Card', ['Yes', 'No'])
isactivemember = st.selectbox('Is Active Member', ['Yes', 'No'])

input_data = pd.DataFrame({
    'CreditScore': [creditscore],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [numofproducts],
    'HasCrCard': [1 if hascrcard == 'Yes' else 0],
    'IsActiveMember': [1 if isactivemember == 'Yes' else 0],
    'EstimatedSalary': [estimatedsalary]
})

geography_encoded = onehot_encoder.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

# Ensure the features are in the same order as used during training
expected_columns = list(scaler.feature_names_in_)
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[expected_columns]

input_data_scaled = scaler.transform(input_data)


prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]
st.write(f'Predicted Salary: {predicted_salary:.2f}')