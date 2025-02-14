import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# Function to fetch real-world weather data (example using Weather API)
def fetch_weather_data(lat, lon, api_key):
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={lat},{lon}&dt=2024-02-01"
    response = requests.get(url)
    data = response.json()
    return data

# Function to fetch soil data (example using ISRIC SoilGrids API)
def fetch_soil_data(lat, lon):
    url = f"https://rest.soilgrids.org/query?lon={lon}&lat={lat}&attributes=phh2o,sand,silt,clay"
    response = requests.get(url)
    data = response.json()
    return data

# Load real-world crop dataset (example from USDA or FAO)
def load_crop_data():
    url = "https://example.com/crop_data.csv"  # Replace with actual URL
    df = pd.read_csv(url)
    return df

# Load data
data = load_crop_data()

# Splitting the data
X = data[['temperature', 'rainfall', 'soil_moisture', 'ph', 'sand', 'silt', 'clay']]
y = data['growth_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Streamlit app
def main():
    st.title("ML-Powered Agronomy Platform")
    
    lat = st.text_input("Latitude")
    lon = st.text_input("Longitude")
    
    if st.button("Fetch Data"):
        weather_data = fetch_weather_data(lat, lon, "your_api_key")
        soil_data = fetch_soil_data(lat, lon)
        
        temp = weather_data['temperature']
        rainfall = weather_data['rainfall']
        soil_moisture = soil_data['moisture']
        ph = soil_data['ph']
        sand = soil_data['sand']
        silt = soil_data['silt']
        clay = soil_data['clay']
        
        prediction = model.predict([[temp, rainfall, soil_moisture, ph, sand, silt, clay]])
        st.write(f"Predicted Crop Growth Rate: {prediction[0]:.2f}")
    
    st.write(f"Model Mean Squared Error: {mse:.2f}")

if __name__ == "__main__":
    main()

# Create Dockerfile
dockerfile_content = """
FROM python:3.9
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
"""
with open("Dockerfile", "w") as f:
    f.write(dockerfile_content)

# Create devcontainer.json
devcontainer_content = """
{
    "name": "ML Agronomy Platform",
    "image": "mcr.microsoft.com/devcontainers/python:3.9",
    "features": {
        "ghcr.io/devcontainers/features/docker-in-docker:1": {}
    },
    "customizations": {
        "vscode": {
            "extensions": ["ms-python.python", "ms-python.vscode-pylance"]
        }
    },
    "postCreateCommand": "pip install -r requirements.txt"
}
"""
with open(".devcontainer/devcontainer.json", "w") as f:
    f.write(devcontainer_content)
