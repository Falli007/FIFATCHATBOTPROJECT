import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.ensemble import AdaBoostClassifier
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from bs4 import BeautifulSoup
import joblib
import requests
import re

heart_data = pd.read_csv(r"C:\Users\allif\Downloads\heart_attack.csv")

# To remove the "fasting blood sugar" column from the training data
heart_data = heart_data.drop("fasting blood sugar", axis=1)


# To load the trained model
random_forest_model = joblib.load("heart_attack_model.joblib")
lgbm_model = joblib.load("trained_lgb_model.joblib")

# To load the scaler
scaler = joblib.load("heart_attack_scaler.joblib")

# To define the columns to scale in the desired order
to_scale = heart_data[["age", "sex", "chest pain type", "resting bp s", "cholesterol", "resting ecg", "max heart rate", "exercise angina", "oldpeak", "ST slope"]]
# Scale the selected columns
df_scaled = scaler.transform(to_scale)
df_scaled = pd.DataFrame(df_scaled, columns=to_scale.columns)

# To append the target column to the scaled dataframe
df_scaled["target"] = heart_data["target"]

# To save the scaler using joblib
joblib.dump(scaler, "heart_attack_scaler.joblib")

def myScaler(data):
    data_ = data.copy()
    if "target" in data_:
        data_ = data_.drop("target", axis=1)  # Drop the "target" column if present
    data_ = data_[to_scale.columns]  # Reorder columns based on the feature order used during scaling
    data_scaled_ = scaler.transform(data_)
    data_scaled_ = pd.DataFrame(data_scaled_, columns=data_.columns)
    data_scaled_.index = data_.index
    return data_scaled_

def myInverseScale(data):
    data_scaler_inverse = scaler.inverse_transform(data)
    data_scaler_inverse = pd.DataFrame(data_scaler_inverse, columns=data.columns)
    data_scaler_inverse.index = data.index
    return data_scaler_inverse

# Function to scrape heart attack information from the website
def scrape_heart_attack_info():
    url = "https://www.nhs.uk/conditions/heart-attack/"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract the required information
    page_content = soup.get_text()

    # Extract overview
    overview_start = page_content.find("A heart attack, or myocardial infarction")
    overview_end = page_content.find("If you think you're having a heart attack")
    overview = page_content[overview_start:overview_end].strip()

    # Extract symptoms
    symptoms_start = page_content.find("Symptoms of a heart attack")
    symptoms_end = page_content.find("Preventing a heart attack")
    symptoms = page_content[symptoms_start:symptoms_end].strip()

    # Extract prevention
    prevention_start = page_content.find("Preventing a heart attack")
    prevention_end = page_content.find("When to seek medical advice")
    prevention = page_content[prevention_start:prevention_end].strip()

    return overview, symptoms, prevention

# Function to save user feedback to a file or database
def save_feedback(feedback):
    # Implement the logic to save feedback to a file or database
    # Appending the feedback to a CSV file
    feedback_df = pd.DataFrame(feedback, columns=["Prediction Accuracy", "Usefulness"])
    feedback_df.to_csv("feedback.csv", index=False)

# Function to make predictions using the selected model
def predict_with_model(model, input_data):
    # Implement prediction logic based on the specific model type
    prediction = model.predict(input_data)
    return prediction
def main():
    st.title("Heart Attack Prediction")

    st.sidebar.title("Sections")
    overview_checkbox = st.sidebar.checkbox("Overview")
    symptoms_checkbox = st.sidebar.checkbox("Symptoms")
    prevention_checkbox = st.sidebar.checkbox("Prevention")

    st.sidebar.title("Instructions on how to use the heart attack prediction app")
    st.sidebar.write("The input contains the following columns:")
    st.sidebar.write("- Age: displays the age of the individual.")
    st.sidebar.write(
        "- Sex: displays the gender of the individual using the following format: 1 = male, 0 = female.")
    st.sidebar.write(
        "- Chest Pain Type: displays the type of chest pain experienced by the individual using the following format: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic.")
    st.sidebar.write(
        "- Resting Blood Pressure: displays the resting blood pressure value of an individual in mmHg.")
    st.sidebar.write("- Serum Cholesterol: displays the serum cholesterol in mg/dL.")
    st.sidebar.write(
        "- Fasting Blood Sugar: compares the fasting blood sugar value of an individual with 120 mg/dL. If fasting blood sugar > 120 mg/dL then 1 (true) else 0 (false).")
    st.sidebar.write(
        "- Resting ECG: 0 = normal, 1 = having ST-T wave abnormality, 2 = left ventricular hypertrophy.")
    st.sidebar.write("- Max Heart Rate Achieved: displays the max heart rate achieved by an individual.")
    st.sidebar.write("- Exercise Induced Angina: 1 = yes, 0 = no.")
    st.sidebar.write("- Oldpeak: oldpeak = ST [Numeric value measured in depression].")
    st.sidebar.write(
        "- ST Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]. 1 = upsloping, 2 = flat, 3 = downsloping.")
    st.sidebar.write("- Target: output class [1: heart attack, 0: Normal].")

    # Add a model selection widget
    selected_model = st.selectbox("Select Model", ["Random Forest", "LightGBM"])

    if selected_model == "Random Forest Classifier":
        model = random_forest_model
    elif selected_model == "lightgbm":
        model = lgb_model

    if overview_checkbox:
        st.subheader("Overview")
        st.write(
            "A heart attack, or myocardial infarction, occurs when the blood supply to the heart muscle is blocked. "
            "The blockage is usually a result of a buildup of plaque in the arteries. "
            "If you suspect a heart attack, call emergency services immediately.")

    if symptoms_checkbox:
       st.subheader("Symptoms")
       st.write("Common symptoms of a heart attack include chest pain or discomfort, shortness of breath, "
               "nausea or vomiting, lightheadedness, and pain or discomfort in the jaw, neck, back, or arms. "
               "It's important to note that not everyone experiences the same symptoms, and some may have no symptoms at all.")

    if prevention_checkbox:
       st.subheader("Prevention")
       st.write(
           "Preventing a heart attack involves adopting a healthy lifestyle. This includes quitting smoking, "
           "eating a balanced diet low in saturated and trans fats, maintaining a healthy weight, "
           "exercising regularly, managing stress, and controlling conditions like high blood pressure, "
           "high cholesterol, and diabetes.")

    st.write("Enter the patient's information:")
    age = st.number_input("Age", min_value=0,
                          help="Enter the age of the individual. Medical advice: Age is a risk factor for heart attacks. The risk increases with age.")
    sex = st.selectbox("Sex", ["0", "1"],
                       help="Select the gender of the individual. Medical advice: Males are generally at a higher risk of heart attacks compared to females.")
    chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                              help="Select the type of chest pain experienced by the individual. Medical advice: Typical angina (chest pain) is a common symptom of a heart attack.")
    ST_slope = st.selectbox("ST_Slope", [1, 2],
                            help="Select the slope of the peak exercise ST segment. Medical advice: Downward sloping ST segment may indicate a higher risk of heart attack.")
    max_heart_rate = st.selectbox("Max Heart Rate", options=[i for i in range(50, 201)],
                                  help="Select the maximum heart rate achieved by the individual. Medical advice: A higher heart rate during exercise may indicate a higher risk of heart attack.")
    oldpeak = st.selectbox("Oldpeak", options=[i / 10 for i in range(0, 51)],
                           help="Select the depression of the ST segment induced by exercise relative to rest. Medical advice: A significant ST depression may indicate a higher risk of heart attack.")
    resting_bp_s = st.selectbox("Resting BP S", options=[i for i in range(90, 241)],
                                help="Select the resting blood pressure value of the individual in mmHg. Medical advice: High blood pressure is a risk factor for heart attacks.")
    cholesterol = st.selectbox("Cholesterol", options=[i for i in range(100, 601)],
                               help="Select the serum cholesterol level in mg/dL. Medical advice: High cholesterol levels are associated with an increased risk of heart attacks.")
    exercise_angina = st.selectbox("Exercise Angina", ["0", "1"],
                                   help="Select whether the individual experiences exercise-induced angina (chest pain). Medical advice: Exercise-induced angina may indicate a higher risk of heart attack.")
    resting_ecg = st.selectbox("Resting ECG", [0, 1, 2],
                               help="Select the resting electrocardiographic results. Medical advice: Abnormal ECG results may indicate an increased risk of heart attack.")

    submit_button = st.button("Predict")

    if submit_button:
        # Preprocess the input data
        input_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "chest pain type": [chest_pain],
            "ST slope": [ST_slope],
            "max heart rate": [max_heart_rate],
            "oldpeak": [oldpeak],
            "resting bp s": [resting_bp_s],
            "cholesterol": [cholesterol],
            "exercise angina": [exercise_angina],
            "resting ecg": [resting_ecg]
        })

        # Scale the input data
        input_data_scaled = myScaler(input_data)

        # Load the selected model based on user's choice
        if selected_model == "Random Forest":
            model = random_forest_model
        elif selected_model == "LightGBM":
            model = lgbm_model


        # To make predictions using the loaded model
        prediction_prob = model.predict_proba(input_data_scaled)  # Returns probability scores for each class
        predicted_class = model.predict(input_data_scaled)

        if predicted_class[0] == 0:
            st.write("The patient is not likely to have a heart attack.")
            st.write(f"Probability of not having a heart attack: {prediction_prob[0][0]:.2f}")
            st.write(f"Probability of having a heart attack: {prediction_prob[0][1]:.2f}")
        else:
            st.write("The patient is likely to have a heart attack.")
            st.write(f"Probability of not having a heart attack: {prediction_prob[0][0]:.2f}")
            st.write(f"Probability of having a heart attack: {prediction_prob[0][1]:.2f}")

        # To display heart attack information
        if overview_checkbox:
            st.subheader("Heart Attack Overview")
            st.write(overview)

        if symptoms_checkbox:
            st.subheader("Heart Attack Symptoms")
            st.write(symptoms)

        if prevention_checkbox:
            st.subheader("Heart Attack Prevention")
            st.write(prevention)

        st.subheader("Feedback and Rating")
        feedback = st.text_area("Provide your feedback on the accuracy of the prediction or user experience.")

        st.subheader("Rating")
        rating = st.slider("Rate the usefulness of the app", min_value=1, max_value=5, step=1)

        submit_button = st.button("Submit Feedback and Rating")

        if submit_button:
            # Store the feedback and rating in a database or file
            with open("feedback.csv", "a") as file:
                file.write(f"{feedback},{rating}\n")

            st.success("Thank you for your feedback and rating!")

        # Warning
        st.sidebar.write("---")
        st.error(
            "This app provides predictions based on a machine learning model and does not replace professional medical advice. Consult a medical doctor for accurate diagnosis and treatment.")

        # Emergency Help Line
        st.sidebar.title("Emergency Help Line")
        st.sidebar.write("Call 999 for immediate assistance.")

if __name__ == "__main__":
    main()
