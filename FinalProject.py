# load packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import streamlit as st
import joblib

ss = pd.DataFrame({
    "sm_li":[1, 0, 1, 0, 1, 0, 1, 0],
    "income": [1, 2, 3, 4, 5, 6, 7, 8],
    "education":[1, 2, 3, 4, 5, 6, 7, 8],
    "parent":[1, 0, 1, 0, 1, 0, 1, 0], # 1 is a parent, 0 is not a parent
    "married":[1, 0, 1, 0, 1, 0, 1, 0], # 1 is married, 0 is not married
    "female":[1, 0, 1, 0, 1, 0, 1, 0], # 1 is female, 0 is not female
    "age": range(1:150)
})

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# Initialize and fit model
lr = LogisticRegression()
lr.fit(X, y)

# Alternatively, load a pre-trained model
# lr = joblib.load("linkedin_model.pkl")

def li_app(income, education, parent, married, female, age):
    # Create person
    person = [
        income,
        education,
        1 if parent else 0,
        1 if married else 0,
        1 if female else 0,
        age
    ]

    # Predict
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
