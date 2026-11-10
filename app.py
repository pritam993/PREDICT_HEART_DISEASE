
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# =====================================================
# 2. LOAD SAVED MODELS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():

    model = joblib.load(BASE_DIR / "lgr_heart.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    columns = joblib.load(BASE_DIR / "columns.pkl")

    return model, scaler, columns


try:
    model, scaler, columns = load_models()

except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()


# =====================================================
# 3. APPLICATION INTERFACE
# =====================================================

st.title("❤️ Heart Disease Prediction System")

st.write(
    "Enter the required health information to get a "
    "machine-learning prediction."
)

st.info(
    "This application is for educational purposes only "
    "and is not a medical diagnosis."
)


# =====================================================
# 4. USER INPUT FORM
# =====================================================

with st.form("heart_prediction_form"):

    st.subheader("Personal Information")

    age = st.number_input(
        "Age", min_value=1, max_value=120, value=45
    )

    sex = st.selectbox("Sex", ["M", "F"])

    st.subheader("Health Measurements")

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=0, max_value=300, value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=0, max_value=1000, value=200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=0, max_value=250, value=150
    )

    oldpeak = st.number_input(
        "Oldpeak (ST depression)",
        min_value=-10.0,
        max_value=20.0,
        value=0.0,
        step=0.1
    )

    st.subheader("Clinical Features")

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["N", "Y"]
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

    submitted = st.form_submit_button(
        "Predict Heart Disease",
        use_container_width=True
    )


# =====================================================
# 5. PREDICTION
# =====================================================

if submitted:

    try:

        # Numerical features

        numeric_features = pd.DataFrame([{
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak
        }])

        # Scale numerical features

        scaled_numeric = scaler.transform(numeric_features)

        numeric_features = pd.DataFrame(
            scaled_numeric,
            columns=numeric_features.columns
        )

        # Prepare input dictionary

        input_data = {
            "Age": numeric_features["Age"].iloc[0],
            "RestingBP": numeric_features["RestingBP"].iloc[0],
            "Cholesterol": numeric_features["Cholesterol"].iloc[0],
            "FastingBS": fasting_bs,
            "MaxHR": numeric_features["MaxHR"].iloc[0],
            "Oldpeak": numeric_features["Oldpeak"].iloc[0]
        }

        # Categorical features

        categories = {
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingECG": resting_ecg,
            "ExerciseAngina": exercise_angina,
            "ST_Slope": st_slope
        }

        for feature, value in categories.items():
            input_data[f"{feature}_{value}"] = 1

        # Match training columns and their order

        input_df = pd.DataFrame([input_data])

        input_df = input_df.reindex(
            columns=columns,
            fill_value=0
        )

        # Predict

        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "The model predicts a possibility "
                "of heart disease."
            )

        else:

            st.success(
                "The model predicts no heart disease "
                "according to the learned classification pattern."
            )

        # Prediction probability

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            class_1_index = list(model.classes_).index(1)

            probability = probabilities[class_1_index] * 100

            st.write(
                f"Model probability for class 1: "
                f"{probability:.2f}%"
            )

        # Display processed input

        with st.expander("View processed model input"):
            st.dataframe(input_df)

    except Exception as e:

        st.error(f"Prediction error: {e}")
