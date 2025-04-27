import streamlit as st
import pandas as pd
import joblib
from model import *
from overall_tuition_model import *

df = loadData()

reg_loaded = joblib.load("reg_model.pkl")

st.title("Predicting College Tuition")
st.write("Team Members: Lucas He, Aden Zhao, Kaitlyn Lu, Rahi Dasgupta")

st.subheader("How will you raise your child?")

childcare = st.radio("Level", options=["Daycare", "Stay-at Home Parent"])

high_school_graduation_year = st.number_input("High School Graduation Year:", min_value=1900, value=2025)
education_level = st.radio("Level", options=["Private", "public in-state", "public out-of-state"])

if education_level == "Private":
    program = "private"
elif education_level == "public in-state":
    program = "public in-state"
elif education_level == "public out-of-state":
    program = "public out-of-state"

if st.button("Predict Cost"):
    tuition = estimate_tuition(program, int(high_school_graduation_year))
    # new_tuition = reg_loaded.predict() # TODO: need to format data properly
    st.caption(f"{tuition}")


# Note: the prediction doesn't work right now since we need to format the data properly before passing it into the model.
# Run this script with `streamlit run frontend.py
