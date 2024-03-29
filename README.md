# JF_B5_GetAround

# Presentation of the repository
- Three Machine Learning projects of the Jedha Bootcamp certification (Data Fullstack) including linear regression, classification and clustering analysis.
- Contact e-mail: [contact](noyer.estelle@gmail.com)

# Studied projects in this repository
- dashboard: project aiming to propose online dashboard compiling data analysis of Get Around dataset.
- api: project aiming to create /predict endpoint on a new API.

# Folder organization
- data:
    - get_around_delay_analysis.xlsx: working dataset for data analysis and dashboarding.
    - get_around_pricing_project.csv: working dataset for machine learning and API development.

- dashoboard:
    - Home.py: Python script for dashboard home page
    - pages folder: Python scripts for dashboard tabs
    - streamlit folder: config.toml file
    - requirements.txt
    - Dockerfile

- api:
    - Dockerfile
    - requirements.txt
    - app.py: Python script for API development
    - model_bestRidge.pkl: saving of the selected model
    - preprocessor.pkl: saving of the data preprocessing step of the selected model

- DAscopethreshold_MLprice.ipynb: Jupyter notebook compiling all the code for data analysis, graphics and machine learning.

# Additional information
- Python version: 3.10.8
- Main used librairies: pandas, scikit-learn, plotly, numpy, streamlit, fastapi, pydantic, uvicorn
