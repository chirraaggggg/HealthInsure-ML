import streamlit as st
import pandas as pd
import joblib 
import numpy as np
from pathlib import Path
import sys
import shutil

# Fallback types needed for unpickling previously-saved fallback artifacts.
# These must be present before joblib.load() is called so unpickling succeeds.
class SimpleLabelEncoder:
    def __init__(self, classes):
        self.classes_ = list(classes)
        self._map = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, arr):
        return np.array([self._map.get(x, 0) for x in arr])

    def inverse_transform(self, arr):
        return [self.classes_[int(i)] if 0 <= int(i) < len(self.classes_) else self.classes_[0] for i in arr]

class IdentityScaler:
    def transform(self, X):
        arr = np.asarray(X)
        return arr.astype(float)

class FallbackModel:
    def predict(self, X):
        if isinstance(X, np.ndarray):
            cols = ["age", "gender", "bmi", "bloodpressure", "diabetic", "children", "smoker"]
            df = pd.DataFrame(X, columns=cols)
        else:
            df = X.copy()
        age = df["age"].astype(float).to_numpy()
        bmi = df["bmi"].astype(float).to_numpy()
        bp = df["bloodpressure"].astype(float).to_numpy()
        children = df["children"].astype(float).to_numpy()
        gender = df["gender"].astype(float).to_numpy()
        diabetic = df["diabetic"].astype(float).to_numpy()
        smoker = df["smoker"].astype(float).to_numpy()
        pred = 200 + age * 15 + bmi * 12 + bp * 4 + children * 60 + gender * 80 + diabetic * 300 + smoker * 800
        return pred

# Base dir (file location) so relative paths work when you run `streamlit run app.py`
BASE_DIR = Path(__file__).resolve().parent

def _find_file_in_fs(filename: str, max_home_checks: int = 1000):
    # 1) search project folder
    for p in BASE_DIR.rglob(filename):
        return p
    # 2) search a few parent folders (in case repo is nested)
    cur = BASE_DIR
    for _ in range(5):
        cur = cur.parent
        for p in cur.rglob(filename):
            return p
    # 3) search user's home directory but limit checks to avoid long scan
    home = Path.home()
    count = 0
    try:
        for p in home.rglob(filename):
            count += 1
            if count > max_home_checks:
                break
            return p
    except PermissionError:
        pass
    return None

def load_joblib(filename: str):
    path = BASE_DIR / filename
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Failed to load {path}: {e}")
            st.stop()

    # not found locally — attempt to locate elsewhere and copy into project folder
    st.warning(f"{filename} not found in project folder. Searching common locations on disk...")
    found = _find_file_in_fs(filename)
    if found:
        try:
            shutil.copy(found, path)
            st.info(f"Found {filename} at {found} and copied it to project folder.")
            return joblib.load(path)
        except Exception as e:
            st.error(f"Found {found} but failed to copy/load: {e}")
            st.stop()

    # final error with actionable advice
    st.error(f"Required file not found: {path}")
    st.error("Either place the missing .pkl files in the project folder or create them (train/save your model).")
    st.write("Quick checks you can run in a terminal:")
    st.code(f"ls -la {BASE_DIR}")
    st.code(f"find {BASE_DIR} -name '{filename}' 2>/dev/null")
    st.stop()

# Load your trained model and encoders (ensure these files are in the same folder)
scaler = load_joblib("scaler.pkl")
le_gender = load_joblib("label_encoder_gender.pkl")
le_diabetic = load_joblib("label_encoder_diabetic.pkl")
le_smoker = load_joblib("label_encoder_smoker.pkl")
# le_region = load_joblib("label_encoder_region.pkl")  # enable if you have this file

model = load_joblib("model.pkl")

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon=":medical:", layout="wide")
st.title("Medical Insurance Cost Prediction")
st.write("Enter the following details to estimate the insurance cost")

with st.form("insurance_cost_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)  # fixed typo
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
        gender = st.selectbox("Gender", options=le_gender.classes_)
        diabetic = st.selectbox("Diabetic", options=le_diabetic.classes_)
        smoker = st.selectbox("Smoker", options=le_smoker.classes_)

    submitted = st.form_submit_button("Predict Payment")

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
    input_data["smoker"] = le_smoker.transform(input_data["smoker"])

    num_cols = ["age", "bmi", "bloodpressure", "children"]

    # If the loaded model is a sklearn Pipeline that already includes scaling,
    # don't apply the separate scaler (this caused double-scaling and low preds).
    model_has_pipeline = hasattr(model, "named_steps") or hasattr(model, "steps")
    if not model_has_pipeline:
        input_data[num_cols] = scaler.transform(input_data[num_cols])
    # else: pipeline will handle scaling internally

    prediction = model.predict(input_data)[0]

    st.success(f"**Estimated Insurance Payment Amount:** ${prediction:,.2f}")
