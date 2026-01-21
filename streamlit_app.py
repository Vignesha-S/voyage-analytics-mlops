import streamlit as st
import pandas as pd
import requests

# -----------------------------
# Config
# -----------------------------
API_BASE_URL = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="Voyage Analytics",
    layout="centered"
)

st.title("🌍 Voyage Analytics Platform")
st.write("End-to-end ML-powered travel analytics system")

# -----------------------------
# Load Hotel Data (for recommendation)
# -----------------------------
@st.cache_data
def load_hotel_data():
    return pd.read_csv("data/hotels.csv")

hotels_df = load_hotel_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
menu = st.sidebar.radio(
    "Navigation",
    [
        "🏨 Hotel Recommendation",
        "✈️ Flight Price Prediction",
        "👤 Gender Classification"
    ]
)

# =====================================================
# 🏨 HOTEL RECOMMENDATION (PRIMARY OBJECTIVE)
# =====================================================
if menu == "🏨 Hotel Recommendation":
    st.header("🏨 Hotel Recommendation System")

    st.write(
        """
        This module provides **hotel recommendations** based on
        city and price similarity using a **content-based approach**.
        """
    )

    st.markdown("### Select Your Preferences")

    city = st.selectbox(
        "Select City",
        sorted(hotels_df["place"].unique())
    )

    budget = st.slider(
        "Preferred Price per Day",
        int(hotels_df["price"].min()),
        int(hotels_df["price"].max()),
        int(hotels_df["price"].median())
    )

    if st.button("Get Recommendations"):
        # -----------------------------
        # Filter by selected city
        # -----------------------------
        filtered = hotels_df[hotels_df["place"] == city].copy()

        # -----------------------------
        # Aggregate to HOTEL LEVEL
        # -----------------------------
        hotel_profile = (
            filtered
            .groupby(["name", "place"], as_index=False)
            .agg({
                "price": "mean",
                "days": "mean",
                "total": "mean"
            })
        )

        # -----------------------------
        # Price similarity (content-based)
        # -----------------------------
        hotel_profile["price_diff"] = abs(
            hotel_profile["price"] - budget
        )

        # -----------------------------
        # Top 5 unique hotels
        # -----------------------------
        recommendations = (
            hotel_profile
            .sort_values("price_diff")
            .head(5)
        )

        st.success("Top 5 Hotel Recommendations")

        st.dataframe(
            recommendations[
                ["name", "place", "days", "price", "total"]
            ].round(2).reset_index(drop=True)
        )

# =====================================================
# ✈️ FLIGHT PRICE PREDICTION
# =====================================================
elif menu == "✈️ Flight Price Prediction":
    st.header("✈️ Flight Price Prediction")

    with st.form("flight_form"):
        origin = st.text_input("From", "Florianopolis (SC)")
        destination = st.text_input("To", "Rio de Janeiro (RJ)")
        flight_type = st.selectbox("Flight Type", ["economic", "premium", "firstClass"])
        agency = st.selectbox("Agency", ["CloudFy", "FlyingDrops", "Rainbow"])
        distance = st.number_input("Distance (km)", min_value=100, value=430)
        time = st.number_input("Flight Time (hrs)", min_value=0.5, value=1.2)
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
        day_of_week = st.selectbox("Day of Week (0=Mon)", list(range(7)))

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        payload = {
            "from": origin,
            "to": destination,
            "flightType": flight_type,
            "agency": agency,
            "distance": distance,
            "time": time,
            "day": day,
            "day_of_week": day_of_week
        }

        response = requests.post(f"{API_BASE_URL}/predict/flight", json=payload)

        if response.status_code == 200:
            price = response.json()["predicted_price"]
            st.success(f"Predicted Flight Price: ₹ {price:.2f}")
        else:
            st.error("API error")

# =====================================================
# 👤 GENDER CLASSIFICATION
# =====================================================
elif menu == "👤 Gender Classification":
    st.header("👤 Gender Classification")

    with st.form("gender_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        company = st.selectbox(
            "Company",
            ["4You", "Acme Factory", "Wonka Company", "Monsters CYA", "Umbrella LTDA"]
        )

        submitted = st.form_submit_button("Predict Gender")

    if submitted:
        payload = {
            "age": age,
            "company": company
        }

        response = requests.post(f"{API_BASE_URL}/predict/gender", json=payload)

        if response.status_code == 200:
            gender = response.json()["predicted_gender"]
            st.success(f"Predicted Gender: {gender.capitalize()}")
        else:
            st.error("API error")
