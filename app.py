# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# ---------- Load Pickle Files Safely ----------
def load_pickle_file(path, description):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"{description} not found at: {path}")
    except Exception as e:
        st.error(f"Failed to load {description}: {e}")
    return None

# Use joblib model instead of .pkl
model = load_pickle_file("rf_model.joblib", "Model")
training_columns = load_pickle_file("training_columns.pkl", "Training Columns")
le_gender = load_pickle_file("le_gender.pkl", "Gender Encoder")
seller_freq = load_pickle_file("seller_freq.pkl", "Seller Frequency Map")
subcat_freq = load_pickle_file("subcat_freq.pkl", "Sub-category Frequency Map")
category_dummy_columns = load_pickle_file("category_dummy_columns.pkl", "Category Dummies")

# ---------- Helper ----------
def create_category_dummies(selected_category):
    dummies = pd.DataFrame([[0]*len(category_dummy_columns)], columns=category_dummy_columns)
    col_name = f'category_{selected_category}'
    if col_name in dummies.columns:
        dummies[col_name] = 1
    return dummies

# ---------- Main App ----------
def main():
    st.set_page_config(page_title="🛍️ Myntra Satisfaction Predictor", layout="centered")
    st.title("🧠 Predict Customer Satisfaction")

    # --- Inputs ---
    price = st.number_input("Price", min_value=0.0, max_value=10000.0, value=1000.0)
    mrp = st.number_input("MRP", min_value=0.0, max_value=15000.0, value=1500.0)
    ratingTotal = st.number_input("Total Ratings", min_value=0, max_value=100000, value=100)

    seller_input = st.text_input("Seller Name (exact match)", value="Roadster")
    seller_encoded = seller_freq.get(seller_input, 0) if seller_freq else 0

    category = st.selectbox("Category", [col.replace("category_", "") for col in category_dummy_columns]) if category_dummy_columns else "Unknown"
    category_dummies_df = create_category_dummies(category)

    subcat_input = st.text_input("Sub-category (exact match)", value="tshirts")
    subcat_encoded = subcat_freq.get(subcat_input, 0) if subcat_freq else 0

    gender = st.selectbox("Gender", ['Male', 'Female', 'Unisex'])
    gender_encoded = le_gender.transform([gender])[0] if le_gender else 0

    discount_percent = st.slider("Discount %", 0, 100, 20)
    is_deep_discount = int(discount_percent >= 50)

    brand_avg_rating = st.slider("Brand Avg Rating", 0.0, 5.0, 4.2)

    # --- Combine All Inputs ---
    input_data = {
        'price': price,
        'mrp': mrp,
        'ratingTotal': ratingTotal,
        'gender': gender_encoded,
        'discount_percent': discount_percent,
        'is_deep_discount': is_deep_discount,
        'brand_avg_rating': brand_avg_rating,
        'seller_freq': seller_encoded,
        'subcat_freq': subcat_encoded
    }
    input_df = pd.DataFrame([input_data])
    full_input_df = pd.concat([input_df.reset_index(drop=True), category_dummies_df], axis=1)

    # Align with training columns
    if training_columns:
        final_input = pd.DataFrame(columns=training_columns)
        for col in training_columns:
            final_input[col] = full_input_df[col] if col in full_input_df.columns else 0
    else:
        final_input = full_input_df

    # --- Prediction ---
    if st.button("Predict Satisfaction"):
        if model:
            try:
                prediction = model.predict(final_input)[0]
                st.success(f"Predicted Satisfaction Category: **{prediction}**")
                if prediction == 0:
                    st.error("Low: Needs better pricing, quality, or branding.")
                elif prediction == 1:
                    st.warning("Medium: Moderate acceptance. Improve for better results.")
                else:
                    st.success("High: Great! Push for promotion and visibility.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("Model not loaded.")

if __name__ == '__main__':
    main()
