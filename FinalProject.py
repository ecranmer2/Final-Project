def li_app (income, education, parent, married, female, age):

    #load packages
    import pandas as pd
    import numpy as np
    import altair as alt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

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
