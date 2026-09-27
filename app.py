import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib

# Project paths
BASE_DIR = Path(__file__).resolve().parent

# Page setup
st.set_page_config(page_title="🏠 House Price Predictor", page_icon="🏡", layout="wide")

# Style
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# Cache model training
@st.cache_resource
def load_resources():
    df = pd.read_csv(BASE_DIR / "house_price_train_dataset.csv")

    if "Distance_km" in df.columns:
        df.rename(columns={"Distance_km":"Distance_from_CityCentre_km"}, inplace=True)

    model = joblib.load(BASE_DIR / "house_price_model.pkl")
    city_encoder = joblib.load(BASE_DIR / "city_encoder.pkl")
    locality_encoder = joblib.load(BASE_DIR / "locality_encoder.pkl")
    r2_scores = joblib.load(BASE_DIR / "r2_scores.pkl")
    features = joblib.load(BASE_DIR / "features.pkl")
    best_model_name = max(r2_scores, key=r2_scores.get)

    return df, model, best_model_name, r2_scores, city_encoder, locality_encoder, features


# Prediction helper
def get_predicted_house_options(city, budget, df, model, city_encoder, locality_encoder, features):
    city_lower = city.strip().lower()
    city_data = df[df['City'].str.lower() == city_lower].copy()
    if city_data.empty:
        return pd.DataFrame()

    try:
        city_data['City_enc'] = city_encoder.transform(city_data['City'])
        city_data['Locality_enc'] = locality_encoder.transform(city_data['Locality'])
    except ValueError:
        return pd.DataFrame()
    X_pred = city_data[features]
    city_data['Predicted_Price'] = model.predict(X_pred)

    min_price = max(0, budget - 3000000)
    max_price = budget + 1000000
    filtered = city_data[
        (city_data['Predicted_Price'] >= min_price) &
        (city_data['Predicted_Price'] <= max_price)
    ]
    if filtered.empty:
        return pd.DataFrame()

    sample_size = min(15, len(filtered))
    sampled = filtered.sample(n=sample_size, random_state=42)
    return sampled[['Locality', 'Distance_from_CityCentre_km', 'Area_sqft', 'BHK',
                    'Land_Price_per_sqft', 'Predicted_Price']]


# Main app
def main():
    st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True)

    with st.spinner("Loading model..."):
        df, best_model, best_model_name, r2_scores, city_encoder, locality_encoder, features = load_resources()
        st.success(f"✅ Models ready! Best: *{best_model_name}*")

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Performance (R²)")
        for name, score in r2_scores.items():
            emoji = "🏆" if name == best_model_name else "📈"
            st.metric(label=f"{emoji} {name}", value=f"{score:.3f}")

        # 📈 Line Graph for Model Accuracy
        st.subheader("📉 Model Accuracy Trend")
        fig_line, ax_line = plt.subplots()
        model_names = list(r2_scores.keys())
        scores = list(r2_scores.values())
        ax_line.plot(model_names, scores, marker='o', linestyle='-', linewidth=2, color='#1f77b4')
        ax_line.set_xlabel("Models")
        ax_line.set_ylabel("R² Score")
        ax_line.set_ylim(0, 1)
        ax_line.set_title("Model Accuracy Comparison")
        for i, v in enumerate(scores):
            ax_line.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)
        st.pyplot(fig_line)

        st.divider()
        st.info("Predict house prices and compare between cities 🚀")

    # --- Main Prediction Section ---
    col1, col2 = st.columns(2)
    with col1:
        city_input = st.selectbox("🏙 Select City", sorted(df['City'].unique()))
    with col2:
        budget_input = st.number_input("💰 Your Budget (INR)", min_value=100000, max_value=100000000, value=5000000, step=100000)

    if st.button("🔍 Find Houses", use_container_width=True):
        options = get_predicted_house_options(city_input, budget_input, df, best_model,
                                              city_encoder, locality_encoder, features)
        if options.empty:
            st.warning("❌ No houses found in this budget range.")
        else:
            st.success(f"🎉 Found {len(options)} options in {city_input}.")
            display_df = options.copy()
            display_df['Predicted_Price'] = display_df['Predicted_Price'].apply(lambda x: f"₹{x:,.0f}")
            display_df['Land_Price_per_sqft'] = display_df['Land_Price_per_sqft'].apply(lambda x: f"₹{x:,.0f}")
            display_df['Area_sqft'] = display_df['Area_sqft'].apply(lambda x: f"{x:,.0f} sqft")
            display_df['Distance_from_CityCentre_km'] = display_df['Distance_from_CityCentre_km'].apply(lambda x: f"{x:.1f} km")

            display_df.columns = ['Locality', 'Distance from Centre', 'Area', 'BHK', 'Land Price/sqft', 'Predicted Price']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- Simplified City Comparison Tool ---
    st.markdown("## 🏙 City Comparison (by Average Price for Selected BHK)")

    cities = sorted(df['City'].unique())
    col1, col2, col3 = st.columns(3)

    with col1:
        city1 = st.selectbox("Select first city", cities, index=0)
    with col2:
        city2 = st.selectbox("Select second city", cities, index=1)
    with col3:
        bhk = st.number_input("Select BHK", min_value=1, max_value=5, value=3)

    if st.button("⚖️ Compare Average Prices", use_container_width=True):
        avg1 = df[(df['City'] == city1) & (df['BHK'] == bhk)]['Total_Price_INR'].mean()
        avg2 = df[(df['City'] == city2) & (df['BHK'] == bhk)]['Total_Price_INR'].mean()

        if pd.isna(avg1) or pd.isna(avg2):
            st.warning("⚠️ Data not available for selected BHK in one or both cities.")
        else:
            comparison_df = pd.DataFrame({
                'City': [city1, city2],
                f'Average Price for {bhk} BHK (INR)': [avg1, avg2]
            })
            comparison_df[f'Average Price for {bhk} BHK (INR)'] = comparison_df[f'Average Price for {bhk} BHK (INR)'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            # Optional bar chart
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar([city1, city2], [avg1, avg2], color=['#1f77b4', '#ff7f0e'])
            ax.set_ylabel("Average Price (INR)")
            ax.set_title(f"Average {bhk} BHK Price Comparison")
            st.pyplot(fig)


if __name__ == "__main__":
    main()


