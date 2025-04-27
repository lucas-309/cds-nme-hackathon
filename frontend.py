import streamlit as st
import pandas as pd
import joblib
from model import *
from overall_tuition_model import *

# hello

reg_loaded = joblib.load("reg_model.pkl")
df = loadData()
private_school_df = pd.read_csv("archive/Private_School_Tuition_by_State.csv")

print(private_school_df)

st.title("The Cost of Tomorrow")
st.subheader("How much will raising a child cost in the future?")
st.write("Team Members: Lucas He, Aden Zhao, Kaitlyn Lu, Rahi Dasgupta")
st.markdown("<br>", unsafe_allow_html=True)

current_year = 2025

birth_year = st.number_input(
    "Select the child's birth year:",
    min_value=current_year,    # Allow selecting a few years back if needed
    value=current_year,            # Default to current year
    step=1
)

inflation_rate = st.slider(
    "Select expected inflation rate (%)",   # Label
    min_value=0.0,                          # Minimum value
    max_value=10.0,                         # Maximum value
    value=2.5,                              # Default value
    step=0.1                                # Step size
)

inflation_factor = (inflation_rate * 0.01 + 1) ** (birth_year - current_year)

tax_deductible = 2000

def inflate(price):
    return round(price * inflation_factor)

st.caption(
    f"Note: the U.S. government provides tax credits of <span style='color: green;'>${2000}</span> per child!",
    unsafe_allow_html=True
)

st.subheader("The Cost of Necessities")

st.caption(f"Housing: <span style='color: red;'>${inflate(5440)}</span>", unsafe_allow_html=True)
st.caption(f"Food: <span style='color: red;'>${inflate(3377)}</span>", unsafe_allow_html=True)
st.caption(f"Transportation: <span style='color: red;'>${inflate(2814)}</span>", unsafe_allow_html=True)
st.caption(f"Healthcare: <span style='color: red;'>${inflate(1688)}</span>", unsafe_allow_html=True)
st.caption(f"Clothing and Miscellaneous: <span style='color: red;'>${inflate(2439)}</span>", unsafe_allow_html=True)
st.caption(f"Childcare: <span style='color: red;'>${inflate(12472)}</span>", unsafe_allow_html=True)

st.caption(f"Total: <span style='color: red;'>${inflate(5440 + 3377 + 2814 + 1688 + 2439 + 12472)}</span>", unsafe_allow_html=True)

st.subheader("The Cost of Adolescence")  
grade_school_type = st.selectbox("Grade School Type:", ["Public", "Private"])
grade_school_state = st.selectbox("Grade School State:", private_school_df["State"].unique())

if grade_school_type == "Public":
    st.caption("Public School: Free!")
else:
    row = private_school_df[private_school_df["State"] == grade_school_state]
    elem_tuition = inflate(row["Elementary School Tuition"].values[0])
    high_tuition = inflate(row["High School Tuition"].values[0])

    st.caption(f"Elementary School Tuition: <span style='color: red;'>${elem_tuition}</span>", unsafe_allow_html=True)
    st.caption(f"High School Tuition: <span style='color: red;'>${high_tuition}</span>", unsafe_allow_html=True)

st.subheader("Nearing Adulthood")

state = st.selectbox("College State:", df["State"].unique())

grad, buffer, level = st.columns([1, 0.5, 1])

college_start_year = birth_year + 18

with grad:
    college_type = st.radio("College Type", options=df["Type"].unique())
with level:
    college_length = st.radio("College Length", options=df["Length"].unique())

st.markdown("<br>", unsafe_allow_html=True)

input = createInput(college_start_year + 4, state, college_type, college_length)
tuition = reg_loaded.predict(input)[0]
tuition = inflate(tuition)  

st.caption(
    f"One year of college will cost <span style='color: red'>${tuition}</span>",
    unsafe_allow_html=True
)



# Run this script with `streamlit run frontend.py
