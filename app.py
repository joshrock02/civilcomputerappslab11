import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# Load the saved preprocessor and model
@st.cache(allow_output_mutation=True)
def load_model_and_preprocessor():
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    model = tf.keras.models.load_model('tf_bridge_model.h5', compile=False)
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

st.title("Bridge Maximum Load Prediction")
st.write("Input the bridge details below to predict its maximum load capacity (in tons).")

# Create input fields for bridge features
span_ft = st.number_input("Span (ft):", min_value=0.0, value=250.0)
deck_width_ft = st.number_input("Deck Width (ft):", min_value=0.0, value=40.0)
age_years = st.number_input("Age (Years):", min_value=0, value=20)
num_lanes = st.number_input("Number of Lanes:", min_value=1, value=4)
condition_rating = st.slider("Condition Rating (1 to 5):", 1, 5, 4)
material = st.selectbox("Material:", options=["Steel", "Concrete", "Composite"])

# Predict the maximum load when the button is pressed
if st.button("Predict Maximum Load"):
    # Create a DataFrame with the input data (the column names must match the training data)
    input_data = pd.DataFrame({
        "Span_ft": [span_ft],
        "Deck_Width_ft": [deck_width_ft],
        "Age_Years": [age_years],
        "Num_Lanes": [num_lanes],
        "Condition_Rating": [condition_rating],
        "Material": [material]
    })
   
    # Preprocess the input data
    input_preprocessed = preprocessor.transform(input_data)
   
    # Get prediction from the model
    prediction = model.predict(input_preprocessed)
   
    st.success(f"Predicted Maximum Load Capacity: {prediction[0][0]:.2f} tons")
