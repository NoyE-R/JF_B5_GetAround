import streamlit as st

# Config
st.set_page_config(
    page_title="Get Around",
    layout="wide"
)

# App
st.title("Get Around: a car rental analysis")
st.markdown("""
    You will find here two kind of analysis performed on dataset provided by Get Around company from their records.

    You could make a round on their web site here: https://fr.getaround.com/.

    Enjoy this simple web dashboard.
    
""")

# Side bar 
st.sidebar.success("What do you want to know?")

# Run the below code if the check is checked
st.subheader("Data analysis")
st.markdown("""
    Second tab of this dashboard is a synthetic data analysis on rental delay and cancelation.

    The dataset focuses on the year 2017 of car rental contracts (ended and canceled) including mobile and connect checkin.

    Purpose of the analysis was to determine the scope of the analysis (connect and/or mobile rentals) and to define a threshold value for the duration between two car rentals.

    The analysis leaded to propose a scope on the **connected** contracts and a threshold value of **240** minutes.
""")

st.subheader("Machine learning model and API")
st.markdown("""
    The third tab of this dashboard displays a short overview of the exploratory analysis and the selected machine learning model.

    The dataset contains car characteristics and daily rental prices.

    The goal for this study was to develop a machine learning model to optimize the daily rental rice offered by the car owners.

    The model can be used thanks to an API that you can find at the following link: https://getaround23fastapi.herokuapp.com/docs.

    Several endpoints are proposed in addition to the prediction option to help you to format your request.
""")