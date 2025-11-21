import streamlit as st
import pandas as pd
import joblib 
import numpy as np

# Load your trained model (make sure model.pkl is in the same folder)
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")
le_diabetic = joblib.load("label_encoder_diabetic.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
le_region = joblib.load("label_encoder_region.pkl")

model = joblib.load("model.pkl")

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon=":medical:", layout="wide")
st.title("Medical Insurance Cost Prediction")
st.write("Enter the following details to estimate the insurance cost")

with st.form("insurance_cost_form"):
    col1, col2 = st.colums(2)
    with col1:
        age = st.number_input("Age", min_value = 0, max_value = 100, value = 30)
        bmi = st.number_input("BMI", min_value= 10.0, max_value=60.0, value= 25.0)
        children = st.number_input("Number of Children", min_value= 0, max_value=0)
    
    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value= 60, max_value= 200, value= 120)
        gender = st.selection("Gender", options = le_gender.classes_)
        diabetic = st.selection("Diabetic", options= le_diabetic.classes_)
        smoker = st.selection("Smoker", options= le_smoker.classes_)

        submitted = st.form_submit_button("Predict Payment")