import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🏠 House Price Prediction App")

st.header("Enter House Details:")

# User Inputs
location = st.selectbox("Location", ['Urban', 'Suburban', 'Rural'])  # Update as per your dataset
housetype = st.selectbox("House Type", ['Apartment', 'Flat', 'Villa'])  # Update as per your dataset
bedroomabvgr = st.number_input("Number of Bedrooms", min_value=0, step=1)
fullbath = st.number_input("Number of Bathrooms", min_value=0, step=1)
lotarea = st.number_input("Lot Area (sq ft)", min_value=0)
garagearea = st.number_input("Garage Area (sq ft)", min_value=0)
firstflrsf = st.number_input("First Floor Area (sq ft)", min_value=0)
yearbuilt = st.number_input("Year Built", min_value=1800, max_value=2050, step=1)

# Predict button
if st.button("Predict Price"):
    # Create input as DataFrame
    input_data = pd.DataFrame([{
        "location": location,
        "housetype": housetype,
        "bedroomabvgr": bedroomabvgr,
        "fullbath": fullbath,
        "lotarea": lotarea,
        "garagearea": garagearea,
        "1stflrsf": firstflrsf,
        "yearbuilt": yearbuilt
    }])

    # Predict using loaded model
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ₹{round(prediction):,}")
