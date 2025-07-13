
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# App title and description
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")
st.title("💓 Heart Attack Risk Predictor")
st.markdown("This app predicts the **likelihood of heart disease** based on clinical inputs using a regression model.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# Input sliders for user data
st.sidebar.header("Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 20, 80, 45)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 202, 150)
    
    sex_num = 1 if sex == "Male" else 0
    data = {
        'age': age,
        'sex': sex_num,
        'cp': cp,
        'chol': chol,
        'thalach': thalach
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Train regression model
X = df[["age", "sex", "cp", "chol", "thalach"]]
y = df["target"]

model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict(input_df)[0]

st.subheader("Prediction Result")
st.write(f"💡 **Predicted risk of heart disease:** {round(prediction * 100)}%")

# Show dataset preview on checkbox
if st.checkbox("Show dataset used for training"):
    st.dataframe(df)
