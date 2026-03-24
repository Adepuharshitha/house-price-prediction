import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("house_model.pkl")

# Load dataset for visualization
data = pd.read_csv("housing.csv")

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction Dashboard")

st.markdown("Predict house prices using Machine Learning")

# Layout
col1, col2 = st.columns(2)

# Input Section
with col1:
    st.subheader("Enter House Details")

    area = st.number_input("Area (sq ft)", 500, 10000, 2000)
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)
    stories = st.number_input("Stories", 1, 5, 1)
    parking = st.number_input("Parking Spaces", 0, 5, 1)

    if st.button("Predict Price"):

        features = np.array([[area, bedrooms, bathrooms, stories, parking]])

        prediction = model.predict(features)

        price = prediction[0]
        lakhs = price / 100000
        crores = price / 10000000

        st.success(f"💰 Estimated Price: ₹ {lakhs:.2f} Lakhs (₹ {crores:.2f} Crore)")

# Visualization Section
with col2:
    st.subheader("📊 Data Visualization")

    fig, ax = plt.subplots()

    ax.scatter(data["area"], data["price"])

    ax.set_xlabel("Area (sq ft)")
    ax.set_ylabel("Price")

    st.pyplot(fig)

# Feature Importance
st.subheader("📊 Feature Importance")

if hasattr(model, "feature_importances_"):

    features = ["Area", "Bedrooms", "Bathrooms", "Stories", "Parking"]

    importance = model.feature_importances_

    fig2, ax2 = plt.subplots()

    ax2.bar(features, importance)

    ax2.set_title("Feature Importance")

    st.pyplot(fig2)

# Dataset Preview
st.subheader("📄 Dataset Preview")

st.dataframe(data.head())