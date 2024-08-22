
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained models and necessary objects
with open('svm_iris_model.pkl', 'rb') as model_file, \
     open('label_encoder.pkl', 'rb') as le_file, \
     open('scaler.pkl', 'rb') as scaler_file:
    model = pickle.load(model_file)
    label_encoder = pickle.load(le_file)
    scaler = pickle.load(scaler_file)

# Streamlit app
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

st.markdown("""
    <div style="background-color:#6a0dad;padding:10px;border-radius:10px">
    <h2 style="color:white;text-align:center;">Iris Flower Species Classification</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("### Enter Iris Flower Measurements")

# User input for measurements
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1)

# Button for prediction
if st.button('Classify'):
    # Prepare the data for prediction
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                              columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    predicted_species = label_encoder.inverse_transform(prediction)
    
    # Display the result
    st.success(f"The predicted species is: **{predicted_species[0]}**")
