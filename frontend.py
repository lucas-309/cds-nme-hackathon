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
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("How will you raise your child?")

grad, buffer, level = st.columns([1, 0.5, 1])

# childcare = st.radio("Level", options=["Daycare", "Stay-at Home Parent"])

state = st.selectbox("State:", df["State"].unique())

with grad:
    high_school_graduation_year = st.number_input("High School Graduation Year:", min_value=2025)
with level:
    education_level = st.radio("Education Level", options=["High School", "College"])

if education_level == "High School":
    tuition = 0
else:
    college_type = st.radio("College Type", options=df["Type"].unique())
    college_length = st.radio("College Length", options=df["Length"].unique())

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Predict Cost"):
    input = createInput(high_school_graduation_year + 4, state, college_type, college_length)
    tuition = reg_loaded.predict(input) # TODO: need to format data properly
    st.caption(f"{tuition}")


# Run this script with `streamlit run frontend.py
