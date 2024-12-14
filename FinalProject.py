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


def new_income(actual_income):
    if actual_income < 10000:
        return 1
    elif 10000 <= actual_income < 20000:
        return 2
    elif 20000 <= actual_income < 30000:
        return 3
    elif 30000 <= actual_income < 40000:
        return 4
    elif 40000 <= actual_income < 50000:
        return 5
    elif 50000 <= actual_income < 75000:
        return 6
    elif 75000 <= actual_income < 100000:
        return 7
    else:
        return 8


def new_education(actual_education):
    if actual_education == "Less than high school":
        return 1
    elif actual_education == "High school incomplete":
        return 2
    elif actual_education == "High school graduate":
        return 3
    elif actual_education == "Some college, no degree":
        return 4
    elif actual_education == "Associate degree":
        return 5
    elif actual_education == "Bachelor's degree":
        return 6
    elif actual_education == "Some postgraduate or professional schooling":
        return 7
    else:
        return 8


def new_parent(actual_parent):
    if actual_parent == "Yes":
        return 1
    else:
        return 0


def new_married(actual_married):
    if actual_married == "Yes":
        return 1
    else:
        return 0


def new_gender(actual_gender):
    if actual_gender == "Female" or actual_gender == "Transgender Female":
        return 1
    else:
        return 0


def li_app(actual_income, actual_education, actual_parent, actual_married, actual_gender, age):
    income = new_income(actual_income)
    education = new_education(actual_education)
    parent = new_parent(actual_parent)
    married = new_married(actual_married)
    female = new_gender(actual_gender)

    # create person
    person = [
        income,
        education,
        parent,
        married,
        female,
        age
    ]

    # predict
    predict = lr.predict([person])
    prob = np.round(lr.predict_proba([person]) * 100, 2)

    return {
        "Predicted Class": "LinkedIn User" if predict[0] == 1 else "Not a LinkedIn User",
        "Probability of LinkedIn User": f"{prob[0][1]}%"
    }


# Streamlit
st.title("LinkedIn User Predictor")
st.write("Enter the details below to find out if someone is likely to be a LinkedIn user.")

# User input
actual_income = st.number_input("Household Income (e.g., $20,000)", min_value=0, step=1000,
                                value=20000)
education_options = ["Less than high school", "High school incomplete", "High school graduate",
                     "Some college, no degree", "Associate degree", "Bachelor's degree",
                     "Some postgraduate or professional schooling", "Graduate degree or higher"]
actual_education = st.selectbox("Education", education_options)
# Parent, Married, Female: Select 1 (Yes) or 0 (No)
actual_parent = st.selectbox("Are you a parent?", options=["Yes", "No"])
actual_married = st.selectbox("Are you married?", options=["Yes", "No"])
actual_gender = st.selectbox("Gender",
                             options=["Male", "Female", "Non-Binary", "Transgender Male", "Transgender Female",
                                      "Prefer Not to Say"])
age = st.number_input("Age", min_value=1, max_value=120, step=1, value=30)

if st.button("Predict"):
    try:
        result = li_app(
            actual_income,  # Pass the actual income entered by the user
            actual_education,  # Pass the actual education level
            actual_parent,  # Pass the actual parent status
            actual_married,  # Pass the actual marital status
            actual_gender,  # Pass the actual gender
            age  # Pass the age
        )
        st.subheader("Prediction Results")
        st.write(f"**Predicted Class:** {result['Predicted Class']}")
        st.write(f"**Probability of LinkedIn User:** {result['Probability of LinkedIn User']}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

