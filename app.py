import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load everything
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("🍄 Mushroom Classification System")

st.write("Enter mushroom characteristics")

# 👇 Categorical inputs (VERY IMPORTANT)
odor = st.selectbox("Odor", ['a','l','c','y','f','m','n','p','s'])
gill_size = st.selectbox("Gill Size", ['b','n'])
stalk_surface_below = st.selectbox("Stalk Surface Below Ring", ['f','y','k','s'])
spore_print = st.selectbox("Spore Print Color", ['k','n','b','h','r','o','u','w','y'])
gill_color = st.selectbox("Gill Color", ['k','n','b','h','g','r','o','p','u','e','w','y'])
ring_type = st.selectbox("Ring Type", ['p','e','l','f','n'])
stalk_surface_above = st.selectbox("Stalk Surface Above Ring", ['f','y','k','s'])
bruises = st.selectbox("Bruises", ['t','f'])

if st.button("Predict"):
    try:
        # Step 1: Create input dictionary
        input_dict = {
            'odor': odor,
            'gill-size': gill_size,
            'stalk-surface-below-ring': stalk_surface_below,
            'spore-print-color': spore_print,
            'gill-color': gill_color,
            'ring-type': ring_type,
            'stalk-surface-above-ring': stalk_surface_above,
            'bruises': bruises
        }

        # Step 2: Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Step 3: One-hot encode
        input_encoded = pd.get_dummies(input_df)

        # Step 4: Align with training columns
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        # Step 5: Scale
        input_scaled = scaler.transform(input_encoded)

        # Step 6: Predict
        prediction = model.predict(input_scaled)

        # Output
        result = "Edible 🍽️" if prediction[0] == 'e' else "Poisonous ☠️"
        st.success(f"Prediction: {result}")

    except Exception as e:
        st.error(f"Error: {str(e)}")