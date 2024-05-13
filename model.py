import streamlit as st 
import pickle
import pandas as pd 

st.title("Heart attack prediction")
st.info("Please fill out the sections below")
st.sidebar.header("Diagnosis")

# Load the trained model
with open("heart.sav", "rb") as file:
    clf = pickle.load(file)

# Input fields
age = st.text_input('age')
trtbps = st.text_input('trtbps')
chol = st.text_input('chol')
thalachh = st.text_input('thalachh')
oldpeak = st.text_input('oldpeak')
sex_1 = st.text_input('sex')
cp_1 = st.text_input('cp')
cp_2 = st.text_input('cp_2')
cp_3 = st.text_input('cp_3')
restecg_1 = st.text_input('restecg_1')
restecg_2 = st.text_input('restecg_2')
exng_1 = st.text_input('exng_1')
slp_1 = st.text_input('slp_1')
slp_2 = st.text_input('slp_2')
caa_1 = st.text_input('caa_1')
caa_2 = st.text_input('caa_2')
thall_2 = st.text_input('thall_2')
thall_3 = st.text_input('thall_3')

# Create a DataFrame with user inputs
dataset = pd.DataFrame({
    'age': [age],
    'trtbps': [trtbps],
    'chol': [chol],
    'thalachh': [thalachh],
    'oldpeak': [oldpeak],
    'sex_1': [sex_1],
    'cp_1': [cp_1],
    'cp_2': [cp_2],
    'cp_3': [cp_3],
    'restecg_1': [restecg_1],
    'restecg_2': [restecg_2],
    'exng_1': [exng_1],
    'slp_1': [slp_1],
    'slp_2': [slp_2],
    'caa_1': [caa_1],
    'caa_2': [caa_2],
    'thall_2': [thall_2],
    'thall_3': [thall_3]
})

# Prediction button
confirm_button = st.sidebar.button('Confirm')
if confirm_button:
    result = clf.predict(dataset)
    st.write(result)
