import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Car Price Predictor", layout="centered")

# Load and clean the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
    data = data.drop_duplicates()

    # Feature engineering
    data['car_age'] = 2025 - data['year']  # derive car's age
    data['brand'] = data['name'].apply(lambda x: x.split()[0])  # extract brand from name
    data['owner'] = data['owner'].replace({
        'Fourth & Above Owner': 'Other',
        'Test Drive Car': 'Other'
    })

    data = data.drop(columns=['name', 'year'])
    return data

# Train the Random Forest model with pipeline
@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']

    # Categorical and numerical columns
    categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
    numerical_features = ['km_driven', 'car_age']

    # One-hot encode categorical variables
    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

    # Build pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline, df

# Load model and data
your_model, car_data = train_model()

# --- UI Styling ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            font-family: 'Segoe UI', sans-serif;
            color: white;
        }
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1502877338535-766e1452684a');
            background-size: cover;
            background-position: center;
        }
        h1, p, label, .stMarkdown {
            color: #ffffff;
        }
        .stButton > button {
            background-color: #ff4b4b;
            color: white;
            padding: 0.6em 1.2em;
            border: none;
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #e63946;
        }
    </style>
""", unsafe_allow_html=True)

# --- Page Layout ---

st.markdown("<h1 style='text-align: center;'>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill out the details below to estimate the car's selling price.</p>", unsafe_allow_html=True)

# Extract options from the data
brands = sorted(car_data['brand'].unique())
fuel_options = car_data['fuel'].unique()
seller_options = car_data['seller_type'].unique()
transmission_options = car_data['transmission'].unique()
owner_options = car_data['owner'].unique()

# Centered input form
left, center, right = st.columns([1, 2, 1])
with center:
    selected_brand = st.selectbox("Car Brand", brands)
    selected_year = st.slider("Year of Manufacture", 1992, 2025, 2015)
    selected_fuel = st.selectbox("Fuel Type", fuel_options)
    selected_seller = st.selectbox("Seller Type", seller_options)
    selected_transmission = st.selectbox("Transmission", transmission_options)
    selected_owner = st.selectbox("Ownership", owner_options)
    entered_kms = st.number_input("Kilometers Driven", min_value=0, value=50000)

    if st.button("Predict Price"):
        # Construct input features
        car_age = 2025 - selected_year
        user_input = pd.DataFrame({
            'fuel': [selected_fuel],
            'seller_type': [selected_seller],
            'transmission': [selected_transmission],
            'owner': [selected_owner],
            'brand': [selected_brand],
            'km_driven': [entered_kms],
            'car_age': [car_age]
        })

        # Predict and show the result
        predicted_price = your_model.predict(user_input)[0]
        formatted_price = f"\u20b9 {int(predicted_price):,}"
        st.success(f"Estimated Selling Price: {formatted_price}")
