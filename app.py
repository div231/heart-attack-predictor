import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# App title
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Risk Predictor")
st.markdown("This app predicts **heart disease risk** using a logistic regression model based on patient data.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# Sidebar input
st.sidebar.header("Enter Patient Data:")

def get_user_input():
    age = st.sidebar.slider("Age", 20, 80, 45)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trtbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.sidebar.slider("Rest ECG (0-2)", 0, 2, 1)
    thalachh = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
    exng = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    slp = st.sidebar.slider("Slope (0-2)", 0, 2, 1)
    caa = st.sidebar.slider("No. of major vessels", 0, 4, 0)
    thall = st.sidebar.slider("Thalassemia (0-2)", 0, 2, 1)

    sex_val = 1 if sex == "Male" else 0

    data = {
        "age": age,
        "sex": sex_val,
        "cp": cp,
        "trtbps": trtbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalachh": thalachh,
        "exng": exng,
        "oldpeak": oldpeak,
        "slp": slp,
        "caa": caa,
        "thall": thall
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# Model training
X = df.drop("output", axis=1)
y = df["output"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Prediction
prediction = model.predict(input_df)[0]
confidence = model.predict_proba(input_df)[0][1] * 100

st.subheader("Prediction Result")
st.write(f"🩺 **Prediction:** {'At Risk 💀' if prediction == 1 else 'Low Risk 💚'}")
st.write(f"📊 **Confidence:** {confidence:.2f}%")

# Show data
if st.checkbox("Show Training Dataset"):
    st.dataframe(df)
