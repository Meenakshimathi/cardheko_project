import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained Random Forest model
model = joblib.load(r'F:\VSCODE\random_forest_model.pkl')

# Load the dataset to get available options for certain fields
df_cars = pd.read_csv('C:/Users/DELL/Desktop/cars3/combined_cl.csv')

# Define the function for predicting price
def predict_price(km, modelYear, car_model, variantName, Color, Displacement, NoDoorNumbers):
    # Create a dictionary with the user input values
    input_data = {
        'km': [km],
        'modelYear': [modelYear],
        'model': [car_model],
        'variantName': [variantName],
        'Color': [Color],
        'Displacement': [Displacement],
        'NoDoorNumbers': [NoDoorNumbers]
    }
    
    # Convert the input dictionary to a DataFrame
    input_df = pd.DataFrame(input_data)
    
    # One-hot encode categorical features to match the model's format
    input_df = pd.get_dummies(input_df)
    
    # Align the input DataFrame columns with the model's training data
    model_columns = model.feature_names_in_
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Predict the price using the model
    predicted_price = model.predict(input_df)[0]
    
    return predicted_price

# Streamlit app layout
st.title("Car Price Prediction App")

# Input fields for user to enter values
st.write("Please enter the car details to predict its price:")

km = st.number_input("Kilometers Driven", min_value=0)
modelYear = st.number_input("Model Year", min_value=1980, max_value=2023)
car_model = st.selectbox("Model", df_cars['model'].unique())
variantName = st.selectbox("Variant Name", df_cars['variantName'].unique())
Color = st.selectbox("Color", df_cars['Color'].unique())
Displacement = st.number_input("Displacement (cc)", min_value=500, max_value=5000)
NoDoorNumbers = st.selectbox("Number of Doors", sorted(df_cars['NoDoorNumbers'].unique()))

# Predict button
if st.button("Predict Price"):
    price = predict_price(km, modelYear, car_model, variantName, Color, Displacement, NoDoorNumbers)
    st.write(f"Predicted Price: ${price:,.2f}")
