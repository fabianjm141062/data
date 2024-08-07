import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data (replace with actual data)
data = pd.DataFrame({
    'soil_layer_1_density': [1.8, 1.6, 2.0],
    'soil_layer_1_depth': [5, 6, 4],
    'soil_layer_2_density': [2.2, 2.1, 2.3],
    'soil_layer_2_depth': [10, 8, 12],
    'soil_layer_3_density': [2.5, 2.4, 2.6],
    'soil_layer_3_depth': [15, 14, 16],
    'pile_length': [25, 28, 24],
    'pile_diameter': [0.5, 0.6, 0.45],
    'bearing_capacity': [2000, 2500, 1800]
})

# Features and target
X = data.drop('bearing_capacity', axis=1)
y = data['bearing_capacity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load pre-trained model (train the model as shown above)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Sample data for demonstration
data = pd.DataFrame({
    'soil_layer_1_density': [1.8, 1.6, 2.0],
    'soil_layer_1_depth': [5, 6, 4],
    'soil_layer_2_density': [2.2, 2.1, 2.3],
    'soil_layer_2_depth': [10, 8, 12],
    'soil_layer_3_density': [2.5, 2.4, 2.6],
    'soil_layer_3_depth': [15, 14, 16],
    'pile_length': [25, 28, 24],
    'pile_diameter': [0.5, 0.6, 0.45],
    'bearing_capacity': [2000, 2500, 1800]
})

# Features and target
X = data.drop('bearing_capacity', axis=1)
y = data['bearing_capacity']

# Train the model
model.fit(X, y)

# Streamlit UI
st.title("Pile Foundation Bearing Capacity Prediction")

# User inputs
soil_layer_1_density = st.number_input("Soil Layer 1 Density (g/cm^3)")
soil_layer_1_depth = st.number_input("Soil Layer 1 Depth (m)")
soil_layer_2_density = st.number_input("Soil Layer 2 Density (g/cm^3)")
soil_layer_2_depth = st.number_input("Soil Layer 2 Depth (m)")
soil_layer_3_density = st.number_input("Soil Layer 3 Density (g/cm^3)")
soil_layer_3_depth = st.number_input("Soil Layer 3 Depth (m)")
pile_length = st.number_input("Pile Length (m)")
pile_diameter = st.number_input("Pile Diameter (m)")

# Prediction
if st.button("Predict"):
    input_data = np.array([[soil_layer_1_density, soil_layer_1_depth, soil_layer_2_density,
                            soil_layer_2_depth, soil_layer_3_density, soil_layer_3_depth,
                            pile_length, pile_diameter]])
    prediction = model.predict(input_data)
    st.write(f"Predicted Bearing Capacity: {prediction[0]:.2f} kN")

