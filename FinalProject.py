# load packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import streamlit as st
import joblib


np.random.seed(42)
ss = pd.DataFrame({
    "sm_li": np.random.choice([0, 1], size=100),
    "income": np.random.randint(1, 9, size=100),
    "education": np.random.randint(1, 9, size=100),
    "parent": np.random.choice([0, 1], size=100),
    "married": np.random.choice([0, 1], size=100),
    "female": np.random.choice([0, 1], size=100),
    "age": np.random.randint(1, 101, size=100),
})

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

lr = LogisticRegression(class_weight='balanced')
lr.fit(X, y)

def li_app(income, education, parent, married, female, age):
    # Create person
    person = [
        income,
        education,
        parent,
        married,
        female,
        age,
    ]

    # Predict
    predict = lr.predict([person])
    prob = np.round(lr.predict_proba([person]) * 100, 2)

    return {
        "Predicted Class": "LinkedIn User" if predict[0] == 1 else "Not a LinkedIn User",
        "Probability of LinkedIn User": f"{prob[0][1]}%",
    }

# Streamlit app
st.title("LinkedIn User Predictor")
st.write("Enter the details below to find out if someone is likely to be a LinkedIn user.")

# User input
income = st.number_input("Income (1–8)", min_value=1, max_value=8, step=1, value=2)
education = st.number_input("Education Level (1–8)", min_value=1, max_value=8, step=1, value=2)

# Parent, Married, Female: Select 1 (Yes) or 0 (No)
parent = st.selectbox("Are you a parent? (1 = Yes, 0 = No)", options=[1, 0])
married = st.selectbox("Are you married? (1 = Yes, 0 = No)", options=[1, 0])
female = st.selectbox("Are you female? (1 = Yes, 0 = No)", options=[1, 0])

age = st.number_input("Age (any number)", min_value=1, max_value=120, step=1, value=30)

# Predict button
if st.button("Predict"):
    try:
        result = li_app(income, education, parent, married, female, age)
        st.subheader("Prediction Results")
        st.write(f"**Predicted Class:** {result['Predicted Class']}")
        st.write(f"**Probability of LinkedIn User:** {result['Probability of LinkedIn User']}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
