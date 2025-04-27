import streamlit as st
import pandas as pd
import joblib
from model import *
from overall_tuition_model import *

# hello

reg_loaded = joblib.load("reg_model.pkl")
df = loadData()

st.title("Predicting College Tuition")
st.write("Team Members: Lucas He, Aden Zhao, Kaitlyn Lu, Rahi Dasgupta")

st.subheader("How will you raise your child?")

childcare = st.radio("Level", options=["Daycare", "Stay-at Home Parent"])

state = st.selectbox("State:", df["State"].unique())

high_school_graduation_year = st.number_input("High School Graduation Year:", min_value=2025)
education_level = st.radio("Education Level", options=["High School", "College"])

if education_level == "High School":
    tuition = 0
else:
    college_type = st.radio("College Type", options=df["Type"].unique())
    college_length = st.radio("College Length", options=df["Length"].unique())

if st.button("Predict Cost"):
    input = createInput(high_school_graduation_year + 4, state, college_type, college_length)
    tuition = reg_loaded.predict(input) # TODO: need to format data properly
    st.caption(f"{tuition}")


# Run this script with `streamlit run frontend.py
