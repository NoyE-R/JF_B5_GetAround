import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np

# Config
st.set_page_config(
    page_title="Get Around",
    layout="wide"
)

# App
st.title("Get Around: a car rental analysis")
st.markdown("""
    You will find here a synthetic data analysis on rental delay and cancelation.

    Data were provided by Get Around company from their records.

    You could make a round on their web site here: https://fr.getaround.com/.

    Enjoy this simple web dashboard.
    
""")

# data
@st.cache
def load_data(nrows):
    xls = pd.ExcelFile('get_around_delay_analysis.xlsx')
    dataset = pd.read_excel(xls, 'rentals_data')
    return dataset

st.subheader("Load and showcase data")
data_load_state = st.text('Loading data...')
dataset = load_data(1000)
data_load_state.text("")

# Side bar 
st.sidebar.success("What do you want to know?")

# Run the below code if the check is checked
st.markdown("""
    Dataset of the year 2017 of car rental contracts (ended and canceled) including mobile and connect checkin.

""")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(dataset)

## overview
st.header("Some numbers")
col1, col2, col3 = st.columns(3)
col1.metric("Rentals number", value=dataset.shape[0])
col2.metric("Number of mobile checkin", value=dataset.loc[dataset["checkin_type"] != "connect",:].shape[0])
col3.metric("Number of connected checkin", value=dataset.loc[dataset["checkin_type"] == "connect",:].shape[0])

st.markdown("""
    Number of canceled and ended car rental contracts by checkin type.

""")

counted = dataset[["state", "checkin_type"]].groupby(["state", "checkin_type"]).value_counts().reset_index()
counted = counted.rename(columns={"checkin_type":"type of checkin", 0:"Number of rentals"})
st.table(counted)