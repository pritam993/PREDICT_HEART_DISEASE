# ❤️ Heart Disease Prediction System

A Machine Learning-based web application that predicts the possibility of heart disease using patient health information. The application is built using **Python, Scikit-learn, and Streamlit**.


## 📌 Project Overview

The Heart Disease Prediction System uses a trained Logistic Regression classification model to analyze health-related input features and predict whether a person may have heart disease.

Users can enter their health information through an interactive web interface and receive an instant prediction.

## ✨ Features

- Interactive and user-friendly Streamlit interface.
- Predicts the possibility of heart disease.
- Accepts personal and clinical health information.
- Uses a trained Machine Learning model.
- Displays prediction probability.
- Provides processed input data for inspection.
- Online deployment support through Streamlit Community Cloud.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Model:** Logistic Regression
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Web Framework:** Streamlit

## 📊 Input Features

The application accepts the following 11 input features:

| Feature | Description |
|---|---|
| Age | Patient's age |
| Sex | Patient's sex |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol level |
| FastingBS | Fasting blood sugar |
| MaxHR | Maximum heart rate |
| Oldpeak | ST depression |
| ChestPainType | Type of chest pain |
| RestingECG | Resting electrocardiogram |
| ExerciseAngina | Exercise-induced angina |
| ST_Slope | Slope of the peak exercise ST segment |

## 📁 Project Structure

```text
Heart_Disease_Prediction/
│
├── app.py
├── lgr_heart.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd Heart_Disease_Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 📈 Prediction Output

The application provides two possible classification outputs:

- **Class 0:** No heart disease predicted by the model.
- **Class 1:** Possibility of heart disease predicted by the model.

The application also displays the model's estimated probability for Class 1 when probability prediction is supported.

## 🌐 Deployment

This application can be deployed using **Streamlit Community Cloud**.

1. Upload the project files to a GitHub repository.
2. Connect your GitHub account to Streamlit Community Cloud.
3. Select the repository and `app.py`.
4. Deploy the application.
5. Share the generated public URL.

## ⚠️ Disclaimer

This application is developed for educational and academic purposes. Its predictions are based on a trained Machine Learning model and should not be considered medical advice or a clinical diagnosis.

Always consult a qualified healthcare professional for medical evaluation.

## 👨‍💻 Author

**Pritam Kundu**

M.Tech in Data Science (2028)

Interested in Machine Learning, Artificial Intelligence, and Data Science.

---

⭐ If you find this project interesting, consider giving the repository a star!
