import streamlit as st

# App
st.set_page_config(
    page_title="API to predict rental prices",
    layout="wide"
)

st.title("Predict rental prices")
st.sidebar.success("What do you want to know?")

st.markdown("""
    You can find the API and its description at the following link: https://getaround23fastapi.herokuapp.com/docs
""")