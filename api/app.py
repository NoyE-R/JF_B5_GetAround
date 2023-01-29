import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI

import pandas as pd
import numpy as np

import pickle

description = """
GetAround pricing optimization API to propose to car owner optimum prices by using a Machine Learning model.

Two sections are available:
## Data preview
* `/preview` get endpoint a few rows of the dataset.
* `/names-columns` get endpoint to access to the name of all columns in the dataset.
* `/type-columns` get endpoint to access to the number of columns containing a given type of variable.
* `/unique-values` get endpoint to access to the unique values for a no numerical columns.

## Machine-Learning
* `/predict` post endpoint proposing a price for the car rental.

More information are available on each endpoint.

For a quick overview of the exploratory data analysis and model performance, you can check the dedicated dashboard at the following link: `https://getaround23.herokuapp.com/`
"""

tags_metadata = [
    {
        "name": "Data-Preview",
        "description": "Endpoints exploring the used dataset"
    },

    {
        "name": "Machine-Learning",
        "description": "Endpoint using Machine Learning model for pricing optimization"
    }
]

# API instance
app = FastAPI(    
    title="Pricing optimization API",
    version="0.1",
    description=description,
    openapi_tags=tags_metadata
)

# class data
class Dataset(BaseModel):
    model_key: str
    mileage: int
    engine_power: int
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

# define endpoints
## welcome message
@app.get("/")
async def index():
    message = "Hello! This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"
    return message

## endpoint preview
@app.get("/preview/{number_row}", tags=["Data-Preview"])
async def preview_dataset(number_row: int = 7):
    """
    Display the a sample of rows of the dataset.
    `number_row` parameter allows to specify the number of rows you would like to display (default value: 10).
    """
    dataset = pd.read_csv("get_around_pricing_project.csv")
    dataset = dataset.drop(axis=1, columns="Unnamed: 0")

    if number_row > len(dataset):
        response = {
            "message": "Dataset has less rows"
        }
    else:
        response = dataset.head(number_row)

    return response.to_json()

## names columns endpoint
@app.get("/names-columns", tags=["Data-Preview"])
async def nameC():
    """
    Get the name of all column of the dataset. No parameter required.

    Expected response:

    {"names of the columns:" : NAMES_COLUMNS}
    """
    dataset = pd.read_csv("get_around_pricing_project.csv")
    dataset = dataset.drop(axis=1, columns="Unnamed: 0")
    
    list_col = {"names of the columns:" : list(dataset.columns)}
    return list_col

## type-columns endpoint
@app.get("/type-columns/{tvar}", tags=["Data-Preview"])
async def uniVal(tvar: str = "numeric"):
    """
    Give the column names for a given type of data.

    You should specify `tvar` parameter by the type of data amnong those options:

    * boolean variable:'bool'
    * numerical variable: 'float64', 'int64' or 'number'
    * string variable: 'object'
    * categorical variable: 'category'
    * datetime variable: 'timedelta'

    Expected response:
    ```
    {
        "type of variable": TYPE_VARIABLE,
        "name of the column" : NAME_COLUMNS
    }
    ```
    """
    dataset = pd.read_csv("get_around_pricing_project.csv")
    dataset = dataset.drop(axis=1, columns="Unnamed: 0")

    type_var = ['bool', 'float64', 'int64', 'number', 'object', 'category', 'timedelta']
    if tvar not in type_var:
        msg = { "message" : f"type not defined correctly"}
        return msg
    else:
        tval = dataset.select_dtypes(include=tvar).columns
        if len(tval) == 0:
            list_val = {"No column of this type in the dataset"}
        else:
            list_val = {
                "type of variable": tvar,
                "name of the column" : list(tval)
                }

        return list_val

## unique-values endpoint
@app.get("/unique-values", tags=["Data-Preview"])
async def unique_values(col: str):
    """
    Get unique values from a given no numerical column.
    
    You should specify `col` parameter by the name of the researched column.

    Expected response:
    ```
    ['VAL1, 'VAL2', ...]
    ```
    """
    dataset = pd.read_csv("get_around_pricing_project.csv")
    dataset = dataset.drop(axis=1, columns="Unnamed: 0")
    df = list(dataset[col].unique())

    return df


## endpoint predict
@app.post("/predict", tags=["Machine-Learning"])
async def predict(my_data: Dataset):
    """
    Prediction for one observation.

    Required all columns values as a dictionary as in the example below:
    ```
    {
        "model_key": "Peugeot",
        "mileage": 20750,
        "engine_power": 50,
        "fuel": "diesel",
        "paint_color": "grey",
        "car_type": "estate",
        "private_parking_available": false,
        "has_gps": true,
        "has_air_conditioning": true,
        "automatic_car": false,
        "has_getaround_connect": true,
        "has_speed_regulator": false,
        "winter_tires": true
    }
    ```
    Be careful for no numerical or boolean variables to fill correctly.
    Get endpoints from Data-Preview can help you to format data.

    /predict endpoint return prediction like:

    {******* Predictions: ***** PREDICTION_VALUE}
    """
    # Read data
    df = pd.DataFrame(dict(my_data), index=[0])
    
    print("\n------ Received values:\n")
    print(df)
    
    # upload model
    with open("model_bestRidge.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    # preprocessing & predict
    to_run = preprocessor.transform(df).toarray()
    prediction = model.predict(to_run)

    # Format response
    response = {"******* Predictions: ***** ": round(prediction[0], 2)}
    return response

# to run locally
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)