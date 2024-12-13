# load packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import streamlit as st
import joblib

s = pd.read_csv("social_media_usage (1).csv")
def clean_sm (x):
    return np.where(x == 1, 1, 0)
ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0), # 1 is a parent, 0 is not a parent
    "married":np.where(s["marital"] == 1, 1, 0), # 1 is married, 0 is not married
    "female":np.where(s["gender"] == 2, 1, np.where(s["gender"] == 1, 0, np.nan)), # 1 is female, 0 is not female
    "age":np.where(s["age"] > 97, np.nan, s["age"])
})
ss = ss.dropna()
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=987)
lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


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
