# load packages
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st
import joblib

def li_app (income, education, parent, married, female, age):

    #create person
    person = [
        income,
        education,
        1 if parent else 0,
        1 if married else 0,
        1 if female else 0,
        age
    ]

    #predict
    predict = lr.predict([person])
    prob = np.round(lr.predict_proba([person]) * 100, 2)

    return {
    "Predicted Class": "LinkedIn User" if predict[0] == 1 else "Not a LinkedIn User",
    "Probability of LinkedIn User": f"{prob[0][1]}%"
    }
# Streamlit app
st.title("LinkedIn User Predictor")
st.write("Enter the details below to find out if someone is likely to be a LinkedIn user.")

# User input fields
income = st.number_input("Income", min_value=0, max_value=8, step=1, value=2)
education = st.number_input("Education Level (e.g., 1 = High School, 2 = Bachelor's, etc.)", min_value=0, max_value=8, step=1, value=2)
parent = st.checkbox("Are you a parent?")
married = st.checkbox("Are you married?")
female = st.checkbox("Are you female?")
age = st.number_input("Age", min_value=0, max_value=100, step=1, value=30)

# Predict button
if st.button("Predict"):
    result = li_app(income, education, parent, married, female, age)
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {result['Predicted Class']}")
    st.write(f"**Probability of LinkedIn User:** {result['Probability of LinkedIn User']}")