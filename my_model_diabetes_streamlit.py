import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

  # Import this for handling images

# Streamlit app configuration
st.set_page_config(page_title='Diabetes Risk Predictor', page_icon=':hospital:', layout='wide')

# Load the trained model
model = load_model(r'C:\Users\Hello\Desktop\kaggle\Project_azure\my_model_final.h5')

# Define the main function
def main():
    
    # Custom CSS to set the background image
    
    st.title('Diabetes Risk Predictor')
    st.markdown("Use the form in the sidebar to input your health indicators and predict the likelihood of diabetes.")
    
    st.markdown(
        """
        ## Author:
        
        ### **Farwa Khalid**
        
        [![GitHub](https://img.shields.io/badge/-GitHub-24292e?style=for-the-badge&logo=github&logoColor=white)](https://github.com/FarwaK05)
        [![Kaggle](https://img.shields.io/badge/-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/farwa99)
        [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077b5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/farwa-khalid-895527280/)
        """
    )

# Call the main function to set up the Streamlit app and display author information and image
main()

# Sidebar for user input
with st.sidebar:
    st.header('Enter Your Information')
    
    HighBP = st.selectbox('High Blood Pressure (0 = No, 1 = Yes)', [0, 1])
    HighChol = st.selectbox('High Cholesterol (0 = No, 1 = Yes)', [0, 1])
    CholCheck = st.selectbox('Cholesterol Check in Past 5 Years (0 = No, 1 = Yes)', [0, 1])
    BMI = st.slider('Body Mass Index (BMI)', 10.0, 50.0, step=0.1)
    Smoker = st.selectbox('Have you smoked at least 100 cigarettes in your entire life? (0 = No, 1 = Yes)', [0, 1])
    Stroke = st.selectbox('Stroke (0 = No, 1 = Yes)', [0, 1])
    HeartDiseaseorAttack = st.selectbox('Heart Disease or Heart Attack (0 = No, 1 = Yes)', [0, 1])
    PhysActivity = st.selectbox('Physical Activity in Past 30 Days (0 = No, 1 = Yes)', [0, 1])
    Fruits = st.selectbox('Consume Fruit 1 or more times per day (0 = No, 1 = Yes)', [0, 1])
    Veggies = st.selectbox('Consume Vegetables 1 or more times per day (0 = No, 1 = Yes)', [0, 1])
    HvyAlcoholConsump = st.selectbox('Heavy Alcohol Consumption (0 = No, 1 = Yes)', [0, 1])
    GenHlth = st.slider('General Health (1 = Excellent, 5 = Poor)', 1, 5)
    MentHlth = st.slider('Mental Health (Number of days with poor mental health past 30 days)', 0, 30)
    PhysHlth = st.slider('Physical Health (Number of days with poor physical health past 30 days)', 0, 30)
    DiffWalk = st.selectbox('Difficulty Walking (0 = No, 1 = Yes)', [0, 1])
    Sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    
    age_categories = {
        1: 'Age 18 - 24',
        2: 'Age 25 - 29',
        3: 'Age 30 - 34',
        4: 'Age 35 - 39',
        5: 'Age 40 - 44',
        6: 'Age 45 - 49',
        7: 'Age 50 - 54',
        8: 'Age 55 - 59',
        9: 'Age 60 - 64',
        10: 'Age 65 - 69',
        11: 'Age 70 - 74',
        12: 'Age 75 - 79',
        13: 'Age 80 or older'
    }
    Age = st.selectbox('Select Your Age Range', options=list(age_categories.keys()), format_func=lambda x: age_categories[x])

# Collect input data for prediction
input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, 
                        Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age]])

# Display input data
st.subheader('User Input Parameters')
input_display_data = {
    'HighBP': HighBP,
    'HighChol': HighChol,
    'CholCheck': CholCheck,
    'BMI': BMI,
    'Smoker': Smoker,
    'Stroke': Stroke,
    'HeartDiseaseorAttack': HeartDiseaseorAttack,
    'PhysActivity': PhysActivity,
    'Fruits': Fruits,
    'Veggies': Veggies,
    'HvyAlcoholConsump': HvyAlcoholConsump,
    'GenHlth': GenHlth,
    'MentHlth': MentHlth,
    'PhysHlth': PhysHlth,
    'DiffWalk': DiffWalk,
    'Sex': Sex,
    'Age': age_categories[Age]  # Display the age range as a string
}
input_df = pd.DataFrame([input_display_data])
st.write(input_df)

# Make prediction
if st.button('Predict Diabetes'):
    with st.spinner('Predicting...'):
        prediction = model.predict(input_data)
        prediction_binary = (prediction > 0.5).astype(int)
        
    st.subheader('Prediction Result')
    if prediction_binary[0][0] == 1:
        st.markdown('### :red[Diabetes Detected]')
    else:
        st.markdown('### :green[No Diabetes Detected]')

    # Display probability
    st.write(f"**Prediction Probability**: {prediction[0][0]:.2f}")

st.write("This app collects user health information to predict diabetes.")

