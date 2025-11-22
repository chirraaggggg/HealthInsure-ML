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

# set page
st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon=":medical:", layout="wide")

# --- UPDATED HEADER + global styles (heading white, improved hero) ---
st.markdown(
    """
    <style>
      /* page background and container tweaks */
      .stApp { background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%); }
      .hero {
        background: linear-gradient(90deg,#0f172a,#0b3d91);
        color: white;
        padding: 28px;
        border-radius: 14px;
        box-shadow: 0 8px 30px rgba(2,6,23,0.12);
        margin-bottom: 18px;
      }
      .hero h1 { margin: 0; font-size: 34px; color: #ffffff; }
      .hero p { margin: 6px 0 0; color: rgba(255,255,255,0.85); }
      .card { background: white; padding: 16px; border-radius: 12px; box-shadow: 0 6px 20px rgba(11,36,84,0.06); }
      .small-muted { color: #6b7280; font-size:12px; }
    </style>

    <div class="hero">
      <div style="display:flex;align-items:center;gap:14px">
        <div style="font-size:44px">🩺</div>
        <div>
          <h1>Medical Insurance Cost Predictor</h1>
          <p>Fast, friendly estimates · Replace with your trained model for production-grade results</p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar Inputs (clean grouped controls) ---
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Patient Information")
    age = st.slider("Age", 18, 90, 35)
    bmi = st.slider("BMI", 15.0, 45.0, 27.5, step=0.1)
    bloodpressure = st.slider("Blood Pressure", 80, 200, 120)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    st.markdown("---")
    st.subheader("Health Status")
    gender = st.selectbox("Gender", options=le_gender.classes_)
    diabetic = st.selectbox("Diabetic", options=le_diabetic.classes_)
    smoker = st.selectbox("Smoker", options=le_smoker.classes_)
    st.markdown("---")
    predict_btn = st.button("Estimate Cost", key="predict_btn", help="Click to compute an insurance cost estimate")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main content: card + details (removed the 'See input summary' expander as requested) ---
result_container = st.container()

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div><strong class='small-muted'>Model:</strong> {type(model).__name__}</div>"
            f"<div class='small-muted'>Tip: Use realistic inputs for best estimates</div>"
            "</div><hr>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if predict_btn:
    with st.spinner("Calculating estimate..."):
        input_data = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "bmi": [bmi],
            "bloodpressure": [bloodpressure],
            "diabetic": [diabetic],
            "children": [children],
            "smoker": [smoker]
        })

        # encode categoricals
        input_data["gender"] = le_gender.transform(input_data["gender"])
        input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
        input_data["smoker"] = le_smoker.transform(input_data["smoker"])

        num_cols = ["age", "bmi", "bloodpressure", "children"]
        model_has_pipeline = hasattr(model, "named_steps") or hasattr(model, "steps")
        if not model_has_pipeline:
            input_data[num_cols] = scaler.transform(input_data[num_cols])

        pred = float(model.predict(input_data)[0])
        pred_rounded = round(pred, 2)

    # Attractive result card
    with result_container:
        left, right = st.columns([2, 1])
        with left:
            st.markdown(
                f"""
                <div style="background:linear-gradient(90deg,#7dd3fc,#60a5fa);padding:20px;border-radius:14px;color:#022c43">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <h3 style="margin:0;color:#001219">Estimated Annual Insurance Cost</h3>
                      <p style="margin:6px 0 0;color:#022c43">Based on provided inputs</p>
                    </div>
                    <div style="text-align:right">
                      <div style="font-size:36px;font-weight:800;color:#001219">${pred_rounded:,.2f}</div>
                      <div style="font-size:12px;color:#023047">Estimated</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # KPIs
            baseline = 12000.0
            diff = pred_rounded - baseline
            pct = (diff / baseline) * 100
            k1, k2, k3 = st.columns(3)
            k1.metric("Estimate", f"${pred_rounded:,.0f}")
            k2.metric("Baseline (avg)", f"${baseline:,.0f}")
            k3.metric("Difference", f"${diff:,.0f}", delta=f"{pct:+.1f}%")
            st.markdown("### Recommendation")
            st.write("- For production accuracy, replace model.pkl with your trained model")
            st.write("- Provide region / medical-history features for better predictions")

        with right:
            st.image("https://cdn-icons-png.flaticon.com/512/2965/2965567.png", width=120)
            st.markdown("<small style='color:gray'>Tip: values are estimates only.</small>", unsafe_allow_html=True)

    st.success(f"Final estimate: ${pred_rounded:,.2f}")

# footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<small style='color:gray'>App powered by your model — update model.pkl for best results.</small>", unsafe_allow_html=True)
