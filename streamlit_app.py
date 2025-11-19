import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and label encoder
model = joblib.load('random_forest_model.joblib')
le = joblib.load('label_encoder.joblib')

# Define the expected features for the model
# These should match the columns of X_train in the training phase
expected_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                     'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic Pressure',
                     'Diastolic Pressure', 'Gender_Male', 'Occupation_Doctor',
                     'Occupation_Engineer', 'Occupation_Lawyer', 'Occupation_Nurse',
                     'Occupation_Sales Representative', 'Occupation_Salesperson',
                     'Occupation_Scientist', 'Occupation_Software Engineer',
                     'Occupation_Teacher', 'BMI Category_Normal Weight',
                     'BMI Category_Obese', 'BMI Category_Overweight']

def predict_sleep_disorder(input_data):
    # Create a DataFrame from input_data
    input_df = pd.DataFrame([input_data])

    # Ensure all expected features are present, and in the correct order
    # Initialize with zeros (or False for bools) for one-hot encoded columns
    processed_input = pd.DataFrame(0, index=[0], columns=expected_features)

    for col in input_df.columns:
        if col in processed_input.columns:
            processed_input[col] = input_df[col].values

    # Make prediction
    prediction_encoded = model.predict(processed_input)

    # Decode the prediction
    prediction_label = le.inverse_transform(prediction_encoded)
    return prediction_label[0]

# Streamlit app interface
st.title('Sleep Disorder Prediction App')
st.write('Enter your details to predict potential sleep disorders.')

# User inputs
with st.form('prediction_form'):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider('Age', 18, 100, 30)
        sleep_duration = st.slider('Sleep Duration (hours)', 4.0, 10.0, 7.0, 0.1)
        quality_of_sleep = st.slider('Quality of Sleep (1-10)', 1, 10, 7)
        physical_activity_level = st.slider('Physical Activity Level (minutes/day)', 0, 150, 60)
        stress_level = st.slider('Stress Level (1-10)', 1, 10, 5)
        heart_rate = st.slider('Heart Rate (bpm)', 50, 100, 70)
        daily_steps = st.slider('Daily Steps', 0, 15000, 7000)
        systolic_pressure = st.slider('Systolic Pressure', 100, 180, 120)
        diastolic_pressure = st.slider('Diastolic Pressure', 60, 120, 80)

    with col2:
        gender = st.selectbox('Gender', ['Female', 'Male'])
        occupation = st.selectbox('Occupation',
                                    ['Accountant', 'Doctor', 'Engineer', 'Lawyer', 'Manager',
                                     'Nurse', 'Sales Representative', 'Salesperson', 'Scientist',
                                     'Software Engineer', 'Teacher'])
        bmi_category = st.selectbox('BMI Category',
                                      ['Normal', 'Normal Weight', 'Overweight', 'Obese'])

    submitted = st.form_submit_button('Predict Sleep Disorder')

    if submitted:
        # Prepare input data for prediction
        input_data = {
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'Physical Activity Level': physical_activity_level,
            'Stress Level': stress_level,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Systolic Pressure': systolic_pressure,
            'Diastolic Pressure': diastolic_pressure
        }

        # Handle one-hot encoding for categorical features
        input_data['Gender_Male'] = 1 if gender == 'Male' else 0

        occupations_ohe = ['Occupation_Doctor', 'Occupation_Engineer', 'Occupation_Lawyer', 
                           'Occupation_Nurse', 'Occupation_Sales Representative', 
                           'Occupation_Salesperson', 'Occupation_Scientist', 
                           'Occupation_Software Engineer', 'Occupation_Teacher']
        for occ_col in occupations_ohe:
            input_data[occ_col] = 1 if occ_col.replace('Occupation_', '') == occupation else 0
            
        # Add 'Occupation_Accountant' and 'Occupation_Manager' if they are not in the list (they might be the base category if not in expected_features)
        # Note: The `expected_features` list defines the columns the model expects. If a category is not in the list, it's implicitly handled by all other categories being 0 (which is the case for `Accountant` and `Manager` based on our `drop_first=True` encoding)
        # For robustness, we check the actual columns present in `expected_features`

        bmi_categories_ohe = ['BMI Category_Normal Weight', 'BMI Category_Obese', 'BMI Category_Overweight']
        for bmi_col in bmi_categories_ohe:
            # Handle 'Normal' as 'Normal Weight' for input consistency
            input_bmi_category = 'Normal Weight' if bmi_category == 'Normal' else bmi_category
            input_data[bmi_col] = 1 if bmi_col.replace('BMI Category_', '') == input_bmi_category else 0

        # Make prediction
        prediction = predict_sleep_disorder(input_data)

        st.success(f'Predicted Sleep Disorder: **{prediction}**')
