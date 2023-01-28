import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# App
st.set_page_config(
    page_title="Predict rental prices",
    layout="wide"
)

st.title("Predict rental prices")
st.sidebar.success("What do you want to know?")

st.markdown("")

## data
@st.cache
def load_data(nrows):
    working = pd.read_csv("./get_around_pricing_project.csv")
    working = working.drop(axis=1, columns="Unnamed: 0")
    return working

working = load_data(None)

st.subheader("Load and showcase data")
data_load_state = st.text('Loading data...')
data_load_state.text("")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(working)

## EDA
st.header("EDA hot points")

### dataset dimensions
st.metric("Number of rows of the dataset", value=working.shape[0])

### missing values
st.markdown("""
    Number of missing values
""")
st.table(100*working.isnull().sum()/working.shape[0])

### visualization
st.markdown("""
    Target (rental_price_per_day) distributions
""")

col1, col2 = st.columns(2, gap="large")
with col1:
    fig1 = px.histogram(data_frame=working, x="rental_price_per_day", color="has_getaround_connect",
                    marginal="rug", hover_data=working.columns,
                    labels={"has_getaround_connect":"Get Around Connected?",
                            "rental_price_per_day":"Rental daily price"},
                    color_discrete_sequence=px.colors.qualitative.T10)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(data_frame=working, x="rental_price_per_day", color="fuel",
                        marginal="rug", hover_data=working.columns,
                        labels={"rental_price_per_day":"Rental daily price"},
                        color_discrete_sequence=px.colors.qualitative.Dark2)
    st.plotly_chart(fig2, use_container_width=True)


st.markdown("""
    Relationships between target (rental_price_per_day) and numerical features
""")

col1, col2 = st.columns(2, gap="large")
with col1:
    fig3 = px.scatter(data_frame=working, x="engine_power", y="rental_price_per_day", color='automatic_car',
                    labels={"engine_power":"Engine power",
                            "rental_price_per_day":"Rental daily price",
                            "automatic_car":"Automatic car"},
                    color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    fig4 = px.scatter(data_frame=working, x="mileage", y="rental_price_per_day", color='winter_tires',
                    labels={"mileage":"Mileage",
                            "rental_price_per_day":"Rental daily price",
                            "winter_tires":"Winter tires"},
                    color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig4, use_container_width=True)

#### correlation matrix
corr_matrix = working.corr().round(2)
fh = go.Figure()
fh.add_trace(
    go.Heatmap(
        x = corr_matrix.columns,
        y = corr_matrix.index,
        z = np.array(corr_matrix),
        text=corr_matrix.values,
        texttemplate='%{text:.2f}'
    )
)

col1, col2, col3 = st.columns([1,5,1])
with col2:
    st.markdown("""
        Correlation matrix between features and target (rental_price_per_day)
    """)
    st.plotly_chart(fh)

## model
st.header("Machine Learning model")
st.markdown("""
    After several tests of linear regression models, the best score (R²) was performed by a **Ridge** model with alpha parameter of 2.

    Scores were computed from **10-fold cross-validation**.

""")
st.markdown("")

### preprocessor and model upload
with open("../src/model_bestRidge.pkl", "rb") as f:
    model = pickle.load(f)

with open("../src/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

target = "rental_price_per_day"
features_list = ['model_key', 'mileage', 'engine_power', 'private_parking_available', 'has_gps', 'fuel', 'paint_color', 'car_type',
                'has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']
numeric_features = ['mileage', 'engine_power']
categorical_features = ['model_key', 'fuel', 'paint_color', 'car_type', 'private_parking_available', 'has_gps',
                        'has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']

X = working.loc[:,features_list]
Y = working.loc[:,target]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

### running preprocessor
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)

### running model
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

scoresR = cross_val_score(model, X_train, Y_train, cv=10)

col1, col2, col3 = st.columns(3)
col1.metric("R² score on training set:", round(r2_score(Y_train, Y_train_pred), 3))
col2.metric("R² score on test set:", round(r2_score(Y_test, Y_test_pred), 3))
col3.metric("R² standard error:", round(scoresR.std(), 3))

### coefficients
st.markdown("")
st.markdown("""
    Contributions of the features to the model performance
""")
column_names = []
for name, pipeline, features_list in preprocessor.transformers_:
    if name == 'num':
        features = features_list
    else: # if pipeline is for categorical variables
        features = pipeline.named_steps['encoder'].get_feature_names_out()
    column_names.extend(features)

coefs = pd.DataFrame(index = column_names, data = model.coef_.transpose(), columns=["coefficients"])
feature_importance = abs(coefs).sort_values(by = 'coefficients')

fig = px.bar(feature_importance, orientation = 'v')
fig.update_layout(showlegend = False, 
                  margin = {'l': 120}
                 )
st.plotly_chart(fig)