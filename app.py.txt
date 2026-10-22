
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Starbucks Beverage Predictor",
    page_icon="☕"
)

st.title("Starbucks Beverage Category Predictor")
st.write("Enter nutritional information to predict the beverage category.")

@st.cache_resource
def load_model():
    return joblib.load("decision_tree_model.pkl")

model = load_model()

features = [
    "Calories",
    "Total Fat (g)",
    "Trans Fat (g)",
    "Saturated Fat (g)",
    "Sodium (mg)",
    "Total Carbohydrates (g)",
    "Cholesterol (mg)",
    "Dietary Fibre (g)",
    "Sugars (g)",
    "Protein (g)",
    "Vitamin A (% DV)",
    "Vitamin C (% DV)",
    "Calcium (% DV)",
    "Iron (% DV)",
    "Caffeine (mg)"
]

with st.form("prediction_form"):
    values = {}

    for feature in features:
        values[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=0.0
        )

    submitted = st.form_submit_button("Predict Category")

if submitted:
    input_data = pd.DataFrame([values])

    try:
        # Model ke training wale columns hasil karein
        model_features = list(model.feature_names_in_)

        # Model ke mutabiq columns tayyar karein
        df_final = pd.DataFrame(
            0, index=[0], columns=model_features
        )

        for feature in features:
            if feature in model_features:
                df_final[feature] = input_data[feature].iloc[0]

            elif feature == "Caffeine (mg)":
                caffeine_value = values[feature]

                for col in model_features:
                    prefix = "Caffeine (mg)_"

                    if col.startswith(prefix):
                        category = col[len(prefix):]

                        try:
                            df_final.loc[0, col] = int(
                                float(category) == float(caffeine_value)
                            )
                        except ValueError:
                            pass

        # Prediction
        prediction = model.predict(df_final)[0]
        st.success(f"Predicted Category: {prediction}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
import pandas as pd
import numpy as np

