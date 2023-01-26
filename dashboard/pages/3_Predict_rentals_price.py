import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score

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
    Relationships between target (rental_price_per_day) and numerical features
""")

col1, col2 = st.columns(2, gap="large")
with col1:
    fig1, ax = plt.subplots(figsize=(3,3))
    sns.set_theme(style="darkgrid")
    sns.scatterplot(data=working, x="mileage", y="rental_price_per_day", legend=False)
    ax.set(xlabel='Mileage',
            ylabel='Daily rental price',
            title='Rental prices = f(Mileage)')
    st.write(fig1)

with col2:
    fig2, ax = plt.subplots(figsize=(3,3))
    sns.set_theme(style="darkgrid")
    sns.scatterplot(data=working, x="engine_power", y="rental_price_per_day", legend=False)
    ax.set(xlabel='Engine power',
            ylabel='Daily rental price',
            title='Rental prices = f(Engine power)')
    st.write(fig2)

#### correlation matrix
corr_matrix = working.corr().round(2)
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(corr_matrix, annot=True, ax=ax)

col1, col2, col3 = st.columns([1,5,1])
with col2:
    st.markdown("""
        Correlation matrix between features and target (rental_price_per_day)
    """)
    st.write(f)

## model
st.header("Machine Learning model")
st.markdown("""
    After several tests of linear regression models, the best score (R²) was performed by a **Ridge** model with alpha parameter of 2.

    Scores were computed from **10-fold cross-validation**.

""")
st.markdown("")

### preprocessing
target = "rental_price_per_day"
features_list = ['model_key', 'mileage', 'engine_power', 'private_parking_available', 'has_gps', 'fuel', 'paint_color', 'car_type',
                'has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']
numeric_features = ['mileage', 'engine_power']
categorical_features = ['model_key', 'fuel', 'paint_color', 'car_type', 'private_parking_available', 'has_gps',
                        'has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']

X = working.loc[:,features_list]
Y = working.loc[:,target]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    # pipeline
    ## for numeric
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

    ## for categorical
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first'))
    ])

    ## column transformer to apply all the transformation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train = preprocessor.fit_transform(X_train).toarray()
X_test = preprocessor.transform(X_test).toarray()

### Perform grid search
best_model = Ridge(alpha=2)
best_model.fit(X_train, Y_train)

scoresR = cross_val_score(best_model, X_train, Y_train, cv=10)

col1, col2, col3 = st.columns(3)
col1.metric("R² score on training set:", round(best_model.score(X_train, Y_train), 3))
col2.metric("R² score on test set:", round(best_model.score(X_test, Y_test), 3))
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

coefs = pd.DataFrame(index = column_names, data = best_model.coef_.transpose(), columns=["coefficients"])
feature_importance = abs(coefs).sort_values(by = 'coefficients')

fig = px.bar(feature_importance, orientation = 'v')
fig.update_layout(showlegend = False, 
                  margin = {'l': 120}
                 )
st.plotly_chart(fig)