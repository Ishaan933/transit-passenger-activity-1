import streamlit as st
import pandas as pd
import joblib 
import numpy as np 
from datetime import datetime

# Dataset URL
DATASET_PATH = "https://drive.google.com/uc?id=1JP1ZUPy7YhzU0YGDCR1y8wp8dkJo5cyX&export=download"

# Model URLs
RF_BOARDINGS_URL = "https://drive.google.com/uc?id=1gUhcvwrsU_8Jzv8t1OfjAKD-Fz-xCARX&export=download"
RF_ALIGHTINGS_URL = "https://drive.google.com/uc?id=1--7a1UQD8WUC4vbQIreJm2tRI3o1ksK1&export=download"
ENCODINGS_URL = "https://drive.google.com/uc?id=1-7hdto0JQvULjoPv_WwRbzAAbcOU6Jsa&export=download"

# Load Dataset
#df = pd.read_parquet(DATASET_URL)

# Load Models
#rf_boardings = joblib.load(RF_BOARDINGS_URL)
#rf_alightings = joblib.load(RF_ALIGHTINGS_URL)
#encodings = joblib.load(ENCODINGS_URL)

# Load dataset, models, and encodings
@st.cache_data
def load_data():
    df = pd.read_parquet(DATASET_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df

@st.cache_resource
def load_models():
    rf_boardings = joblib.load(RF_BOARDINGS_URL)
    rf_alightings = joblib.load(RF_ALIGHTINGS_URL)
    encodings = joblib.load(ENCODINGS_URL)
    return rf_boardings, rf_alightings, encodings

df = load_data()
rf_boardings, rf_alightings, encodings = load_models()

# Sidebar inputs
st.sidebar.header("Passenger Activity Prediction")

# Filter stop numbers
stop_numbers_to_filter = [10545, 10580, 10706, 10619, 10132, 10134, 10707]
filtered_stop_numbers = df['stop_number'].unique()
filtered_stop_numbers = filtered_stop_numbers[np.isin(filtered_stop_numbers, stop_numbers_to_filter)]

stop_number = st.sidebar.selectbox("Select Stop Number", filtered_stop_numbers)

# Filter DataFrame for selected stop number
filtered_df = df[df['stop_number'] == stop_number]

# Get route numbers for the selected stop number
route_numbers = filtered_df['route_number'].unique()
route_number = st.sidebar.selectbox("Select Route Number", route_numbers)

# Get route names for the selected route number
filtered_routes_df = filtered_df[filtered_df['route_number'] == route_number]
route_names = filtered_routes_df['route_name'].unique()
route_name = st.sidebar.selectbox("Select Route Name", route_names)

# Day type and time period inputs
day_type = st.sidebar.selectbox("Select Day Type", ["Weekday", "Saturday", "Sunday"])
time_period = st.sidebar.selectbox(
    "Select Time Period", ["Morning", "Mid-Day", "PM Peak", "Evening", "Night"]
)

# Prediction Logic
def predict_passenger_activity():
    input_data = pd.DataFrame({
        "stop_number": [stop_number],
        "route_number": [encodings["route_number"].get(route_number, -1)],
        "route_name": [encodings["route_name"].get(route_name, -1)],
        "day_type": [encodings["day_type"].get(day_type, -1)],
        "time_period": [encodings["time_period"].get(time_period, -1)],
    })

    # Predictions
    boardings_prediction = rf_boardings.predict(input_data)[0]
    alightings_prediction = rf_alightings.predict(input_data)[0]

    return {
        "boardings_prediction": boardings_prediction,
        "alightings_prediction": alightings_prediction,
    }

# Run Prediction
if st.sidebar.button("Predict"):
    result = predict_passenger_activity()

    st.subheader("Prediction Results")
    st.write(f"**Average Boardings Prediction:** {result['boardings_prediction']:.2f}")
    st.write(f"**Average Alightings Prediction:** {result['alightings_prediction']:.2f}")
