import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import mlflow
import json

import pandas as pd
import numpy as np

description = """
GetAround pricing optimization API to propose to car owner optimum prices by using a Machine Learning model.

Two sections are available:
## Data preview 
* `/preview` get endpoint a few rows of the dataset.
* `/names-columns` get endpoint to access to the name of all columns in the dataset.
* `/type-columns` get endpoint to access to the type of variable is contained in a specific columns.

## Machine-Learning
* `/predict` post endpoint proposing a price for the car rental.

More information are available on each endpoint. 
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
    private_parking: bool
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

## nqmes columns endpoint
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

## unique-values endpoint
@app.get("/type-columns/{col}", tags=["Data-Preview"])
async def uniVal(col: str = "model_key"):
    """
    Give the type for a given column.
    You should specify `col` parameter by the name of the researched column (string).

    Expected response:

    {"type of the column" : TYPE_COLUMN}
    """
    dataset = pd.read_csv("get_around_pricing_project.csv")
    dataset = dataset.drop(axis=1, columns="Unnamed: 0")

    if col not in dataset.columns:
        msg = { "message" : f"Column not defined correctly, please try one of the names: {dataset.columns}"}
        return msg
    else:
        tval = type(dataset[col])
        list_val = {"type of the column" : tval}
        return list_val

## endpoint predict
@app.post("/predict", tags=["Machine-Learning"])
async def predict(my_data: Dataset):
    """
    Prediction for one observation.
    Required all columns values as a dictionary as in the example below:

    {
    "model_key": "Peugeot",
    "mileage": 20750,
    "engine_power": 50,
    "fuel": "diesel",
    "paint_color": "green",
    "car_type": "sport",
    "private_parking": false,
    "has_gps": true,
    "has_air_conditioning": true,
    "automatic_car": false,
    "has_getaround_connect": true,
    "has_speed_regulator": false,
    "winter_tires": true
    }
    
    /predict endpoint return prediction like:

    {'price prediction': PREDICTION_VALUE}
    """
    # Read data
    df = pd.DataFrame(my_data, index=[0])
    
    print("Received values:\n")
    print(df)
    
    # Log model from mlflow 
    logged_model = 'runs:/2b8c72d62a924d82bfbc791a88a8e773/princing_optimization' 

    # Load model as a PyFuncModel & predict
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    prediction = loaded_model.predict(df)

    # Format response
    response = {"price prediction": prediction}
    return response

# to run locally
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)