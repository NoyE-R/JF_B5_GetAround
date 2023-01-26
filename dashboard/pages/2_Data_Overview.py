#data_overview
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Data overview 2017 rentals",
    layout="wide"
)

# App
st.title("Data overview 2017 rentals")
st.sidebar.success("What do you want to know?")

st.markdown("")

## data
@st.cache
def load_data(nrows):
    xls = pd.ExcelFile('./get_around_delay_analysis.xlsx')
    dataset = pd.read_excel(xls, 'rentals_data')
    return dataset

dataset = load_data(None)

## overview
st.header("Metrics")

per_can_con = dataset.loc[(dataset["checkin_type"] == "connect") & (dataset["state"] == "canceled"),:].shape[0] / dataset.loc[dataset["checkin_type"] == "connect",:].shape[0]
per_can_mob = dataset.loc[(dataset["checkin_type"] == "mobile") & (dataset["state"] == "canceled"),:].shape[0] / dataset.loc[dataset["checkin_type"] == "mobile",:].shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("Rentals number", value=dataset.shape[0])
col2.metric("Canceled rentals of mobile checkin", value=str(round(per_can_mob*100, 2))+"%")
col3.metric("Canceled rentals of connected checkin", value=str(round(per_can_con*100, 2))+"%")

## graphs all data
st.header("Duration between rentals")
col1, col2 = st.columns([1, 2])
with col1:
    fig1 = sns.catplot(x="checkin_type", y="time_delta_with_previous_rental_in_minutes", data=dataset, kind="violin")
    fig1.set(xlabel='Type of checkin',
        ylabel='Duration between rentals (min)',
        title='Difference between checkout and next checkin per checkin type')
    st.pyplot(fig1)

with col2:
    fig2 = px.scatter(x="time_delta_with_previous_rental_in_minutes", y="delay_at_checkout_in_minutes", data_frame=dataset,
            title='Is the observed checkout delay depends <br> on the time between rentals?',
            labels={"time_delta_with_previous_rental_in_minutes":"Duration between rentals (min)",
                    "delay_at_checkout_in_minutes":"Checkout delay (min)",
                    "checkin_type":"Checking type"},
            range_y=(-2500, 10000),
            color="checkin_type", height=500, width=700)
    st.plotly_chart(fig2)

tab_mean = dataset[["checkin_type","state", "delay_at_checkout_in_minutes", "time_delta_with_previous_rental_in_minutes"]].groupby(["checkin_type", "state"]).mean()
st.table(tab_mean)

st.markdown("""
    * Values from mobile checkin are more fluctuent and display a lot of outliers, likely due to the client oversights.
    * This means that Mobile checkin are not as reliable that Connect checkin.
    * In the following analysis, I focused on the Connect checkin. However, for Connect checkins, in the case of canceled booking, data about checkout delay are not recorded which limited the analysis.
""")