import streamlit as st
import numpy as np
import pickle

try:
    with open('model/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found! Ensure 'best_model.pkl' exists.")
    st.stop()

st.set_page_config(
    page_title="Medical Insurance Prediction",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Medical Insurance Prediction")
st.markdown("### Predict your maximum insurance charges based on inputs like BMI, age, children, smoker status, sex, and region.")

st.sidebar.header("Input Parameters")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25,
    step=1,
    help="Enter your age (18 - 100).",
)

bmi = st.sidebar.slider(
    "BMI",
    min_value=10.0,
    max_value=50.0,
    value=25.0,
    step=0.1,
    help="Enter your Body Mass Index (10.0 - 50.0).",
)

children = st.sidebar.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
    help="Enter the number of children (0 - 10).",
)

smoker = st.sidebar.radio(
    "Smoker",
    options=["Yes", "No"],
    help="Are you a smoker?",
)
smoker_binary = 1 if smoker == "Yes" else 0

sex = st.sidebar.selectbox(
    "Sex",
    options=["Male", "Female"],
    help="Select your sex.",
)

region = st.sidebar.selectbox(
    "Region",
    options=["northeast", "northwest", "southeast", "southwest"],
    help="Select the region where you live.",
)

sex_binary = 1 if sex == "Male" else 0
region_mapping = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region_encoded = region_mapping[region]

if st.sidebar.button("Predict"):
    input_data = np.array([[age, bmi, children, smoker_binary, sex_binary, region_encoded]])
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Insurance Claimable: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.markdown("### Model Details and Insights")
st.write(
    """
    - **Preprocessing Steps:** Applied label encoding for categorical variables.
    - **Feature Selection:** Used Age, BMI, Children, Smoker Status, Sex, and Region for predictions.
    - **Model Selection:** Utilized Cross-Validation and Grid Search for hyperparameter tuning.
    - **Performance Optimization:** Fine-tuned the model to improve accuracy and reduce overfitting.
    - Predicted charges are based on historical data and may not reflect exact values.
    """
)

st.markdown("---")
st.markdown("Developed by Bhuvanesh P. Powered by Streamlit. 🚀")
