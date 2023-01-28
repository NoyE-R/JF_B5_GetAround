#data_overview
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Data analysis on 2017 rental contracts",
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
st.header("Prospection over all dataset")

st.subheader("Load and showcase data")
data_load_state = st.text('Loading data...')
dataset = load_data(1000)
data_load_state.text("")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(dataset)

## graphs all data
st.subheader("Data overview")

st.markdown("**Some numbers**")
col1, col2, col3 = st.columns(3)
col1.metric("Rentals number", value=dataset.shape[0])
col2.metric("Number of mobile checkin", value=dataset.loc[dataset["checkin_type"] != "connect",:].shape[0])
col3.metric("Number of connected checkin", value=dataset.loc[dataset["checkin_type"] == "connect",:].shape[0])

per_can     = round(dataset.loc[dataset["state"] == "canceled",:].shape[0] / dataset.shape[0] *100, 2)
per_can_con = round(dataset.loc[(dataset["checkin_type"] == "connect") & (dataset["state"] == "canceled"),:].shape[0] / dataset.loc[dataset["checkin_type"] == "connect",:].shape[0] *100, 2)
per_can_mob = round(dataset.loc[(dataset["checkin_type"] == "mobile") & (dataset["state"] == "canceled"),:].shape[0] / dataset.loc[dataset["checkin_type"] == "mobile",:].shape[0] *100, 2)

col1, col2, col3 = st.columns(3)
col1.metric("Canceled rentals", value=str(per_can)+"%")
col2.metric("Canceled rentals of mobile checkin", value=str(per_can_mob)+" %")
col3.metric("Canceled rentals of connected checkin", value=str(per_can_con)+" %")


counted = dataset[["state", "checkin_type"]].groupby(["state", "checkin_type"]).value_counts().reset_index()
counted = counted.rename(columns={"checkin_type":"type of checkin", 0:"Number of rentals"})

col1, col2 = st.columns(2)
with col1:
    fig1 = px.bar(counted, x="type of checkin", y="Number of rentals",
                title="Number of rentals per type of checkin and rental state", text_auto='.2s',
                color='state', barmode='group', color_discrete_sequence=px.colors.qualitative.Dark2)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("")
    st.markdown("")
    st.markdown("""
        **Average values per type of checkin and rental states**
    """)

    tab_mean = dataset[["checkin_type","state", "delay_at_checkout_in_minutes", "time_delta_with_previous_rental_in_minutes"]].groupby(["checkin_type", "state"]).mean().reset_index()
    tab_mean = tab_mean.rename(columns={
        "checkin_type":"type of checkin",
        "delay_at_checkout_in_minutes":"Checkout delay (min)",
        "time_delta_with_previous_rental_in_minutes":"Duration between rentals (min)"
    })
    st.table(tab_mean)

col1, col2 = st.columns(2)
with col1:
    max_cd = round((dataset["delay_at_checkout_in_minutes"].max()/60)/24, 2)
    min_cd = round((dataset["delay_at_checkout_in_minutes"].min()/60)/24, 2)
    type_max = dataset.loc[dataset["delay_at_checkout_in_minutes"]==dataset["delay_at_checkout_in_minutes"].max(),"checkin_type"].values[0]
    type_min = dataset.loc[dataset["delay_at_checkout_in_minutes"]==dataset["delay_at_checkout_in_minutes"].min(),"checkin_type"].values[0]
    mob_can = dataset.loc[(dataset["checkin_type"] == "mobile") & (dataset["state"] == "canceled"),"delay_at_checkout_in_minutes"].unique()

    st.markdown("")
    st.markdown("") 
    st.markdown("**Recorded checkout delay**")
    st.metric("Maximum", value=str(round(max_cd, 2))+" days")
    st.metric("Minimum", value=str(round(min_cd, 2))+" days")

    st.markdown("Both extreme values were identified in mobile checkin records")
    st.markdown("")
    st.markdown("Unique values recorded for canceled rentals in mobile checkin:")
    st.markdown(mob_can)

with col2:
    fig2 = px.box(data_frame=dataset, x="checkin_type", y="delay_at_checkout_in_minutes",
                color_discrete_sequence=px.colors.qualitative.Dark2,
                title="Checkout delay distribution of ended rentals per type of checkin",
                labels={"delay_at_checkout_in_minutes":"Checkout delay (min)",
                        "checkin_type":"Type of checkin"})
    st.plotly_chart(fig2, use_container_width=True)


fig3 = px.violin(x="state", y="time_delta_with_previous_rental_in_minutes",
                box=True, color="state", data_frame=dataset,
                facet_col="checkin_type",
                color_discrete_sequence=px.colors.qualitative.Dark2,
                title="Does duration between rentals impact the rental status?",
                labels={"time_delta_with_previous_rental_in_minutes":"Duration between rentals (min)",
                        "state":"Status of the rental"})
st.plotly_chart(fig3, use_container_width=True)

st.subheader("First conclusion: Scope selection")
st.markdown("""
    * In the case of canceled bookings, data about checkout delay are not recorded as the rentals did not happen.

    * Checkout delay values from mobile checkin are more variable and display outliers, likely due to the client oversights.
    * This means that Mobile checkin are not as reliable that Connect checkin.
    
    * Average duration between rentals are higher for canceled rentals in both checkin types.

    * As the observed trend between ended and canceled rentals are similar between checkin type and the percentage of canceled rentals are in the same range for both checkin, the following analysis focused the Connect checkin.

""")

## graph connect contracts
st.header("Focus on the Connect checkin")
st.subheader("Duration between rentals numbers")

connect = dataset.loc[dataset["checkin_type"] == "connect", :]

col1, col2 = st.columns(2)
with col1:
    med_con = connect["time_delta_with_previous_rental_in_minutes"].quantile()

    st.markdown("")
    st.metric("Median of duration between rental for connected checkin", value=str(round(med_con, 2))+" min")
    
    st.markdown("""
        **Median of duration between rentals per rentals status**        
    """)
    st.markdown("")

    tab_med = connect[["time_delta_with_previous_rental_in_minutes", "state"]].groupby(["state"]).quantile().reset_index()
    tab_med = tab_med.rename(columns={
        "time_delta_with_previous_rental_in_minutes":"Median of the duration between rentals (min)"
    })
    st.table(tab_med)

with col2:
    per_end = round(connect.loc[(connect["time_delta_with_previous_rental_in_minutes"]>240) & (connect["state"] == "ended"),:].shape[0] / connect.shape[0] * 100, 2)
    per_can = round(connect.loc[(connect["time_delta_with_previous_rental_in_minutes"]<240) & (connect["state"] == "canceled"),:].shape[0] / connect.shape[0] * 100, 2)

    st.markdown("""
        **Percentage of bookings above or below the median value of duration between rentals of canceled rentals:**        
    """)
    st.markdown("")
    st.metric("Ended contracts above 240 min", value=str(per_end)+" %")
    st.metric("Canceled contracts below 240 min", value=str(per_can)+" %")

### checkout delay
st.subheader("Does the checkout delay value of the previous rental infer on the rental status?")

connect["delay_previous"] = 1500
prev0 = list(connect["previous_ended_rental_id"].unique())
prev = [item for item in prev0 if not(pd.isna(item)) == True]
for p in prev:
    rent = connect.loc[connect["previous_ended_rental_id"] == p, "rental_id"].values[0]
    delay = connect.loc[connect["rental_id"] == p, "delay_at_checkout_in_minutes"].values

    try:
        float(delay)
        connect.loc[connect["rental_id"] == rent, "delay_previous"] = delay
    except:
        pass

delay = connect.loc[connect["delay_previous"] != 1500,:]

st.markdown(f"""
    Number of data with recorded delay from previous rental: {delay.shape[0]}.
""")

col1, col2 = st.columns(2)
with col1:
    fig4 = px.violin(x="state", y="delay_previous",
            box=True, color="state", data_frame=delay,
            color_discrete_sequence=px.colors.qualitative.Dark2,
            title="Does the checkout delay of the previous rental impact <br> the rental status?",
            labels={"delay_previous":"Checkout delay (min)",
                    "state":"Status of the rental"})
    st.plotly_chart(fig4, use_container_width=True)

    medD_end = delay.loc[delay["state"] == "canceled", "delay_previous"].quantile()
    medD_can = delay.loc[delay["state"] == "ended", "delay_previous"].quantile()

    st.metric("Median value for ended rentals", value=str(medD_end)+" min")
    st.metric("Median value for canceled rentals", value=str(medD_can)+" min")


with col2:
    fig5 = px.scatter(x="time_delta_with_previous_rental_in_minutes", y="delay_previous",
                data_frame=delay, color="state",
                color_discrete_sequence=px.colors.qualitative.Dark2,
                title='Is the observed checkout delay of the previous rental <br> depends on the time between rentals?',
                labels={"time_delta_with_previous_rental_in_minutes":"Duration between rentals (min)",
                        "delay_previous":"Checkout delay (min)"
                        }
                )
    fig5.add_vline(x=delay["time_delta_with_previous_rental_in_minutes"].quantile(),
                    line_width=2, line_dash="dot",
                    annotation_text="duration median")
    st.plotly_chart(fig5, use_container_width=True)

    med_dur = delay["time_delta_with_previous_rental_in_minutes"].quantile()
    st.metric("Median value of the duration between rentals", value=str(med_dur)+" min")

st.subheader("Second conclusion: Threshold value selection")
st.markdown(f"""
    * Checkout delay of the previous rental is more distributed toward low and negative values for ended contracts in comparision to canceled rentals.
    * However, checkout delay of the previous rental does not seem to be correlated to the duration between rentals meaning that the threshold value definition should not be defined regarding the checkout delay value of the previous rental.
    * To avoid half of the canceled rentals, a duration between rentals of **240** min could reduce the percentage of cancelation to {per_can} % instead of {per_can_con} % with a lost of {per_end} % of ended rentals.
""")