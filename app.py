import streamlit as st
import pandas as pd
import joblib 
import numpy as np

# Load your trained model and encoders (ensure these files are in the same folder)
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")
le_diabetic = joblib.load("label_encoder_diabetic.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
le_region = joblib.load("label_encoder_region.pkl")  # This was loaded, but not used in the inputs (add to inputs if required)

model = joblib.load("model.pkl")

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon=":medical:", layout="wide")
st.title("Medical Insurance Cost Prediction")
st.write("Enter the following details to estimate the insurance cost")

with st.form("insurance_cost_form"):
    col1, col2 = st.columns(2)  # Corrected typo: st.columns instead of st.colums
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)  # Increased max_value

    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
        gender = st.selectbox("Gender", options=le_gender.classes_)  # Use st.selectbox
        diabetic = st.selectbox("Diabetic", options=le_diabetic.classes_)
        smoker = st.selectbox("Smoker", options=le_smoker.classes_)

    submitted = st.form_submit_button("Predict Payment")  # Place outside of columns but inside form

if submitted:
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "diabetic": [diabetic],
        "children": [children],
        "smoker": [smoker]
    })

    # Transform categorical columns with loaded encoders
    input_data["gender"] = le_gender.transform(input_data["gender"])
    input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
    # input_data["smoker"] = le_smoker.transform(input_data["smoker"])
    # # If your model expects 'region', you should add region input and transform with le_region too

    # num_cols = ["age", "bmi", "bloodpressure", "children"]
    # input_data[num_cols] = scaler.transform(input_data[num_cols])

    # prediction = model.predict(input_data)[0]

    # st.success(f"**Estimated Insurance Payment Amount:** ${prediction:,.2f}")
