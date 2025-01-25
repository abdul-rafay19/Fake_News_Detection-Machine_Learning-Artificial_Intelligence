import streamlit as st
import joblib
import numpy as np
import time

# Custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            font-family: 'Arial';
            color: #4CAF50;
            text-align: center;
            font-size: 3em;
            margin-bottom: 10px;
        }
        .sub-header {
            text-align: center;
            font-size: 1.5em;
            color: #555;
            margin-bottom: 20px;
        }
        .stTextArea > label {
            font-size: 1.2em;
            color: #4CAF50;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 1.1em;
            border-radius: 8px;
            padding: 10px;
        }
        .stSpinner > div > div {
            color: #4CAF50;
        }
        .result-success {
            background-color: #E8F5E9;
            border-left: 5px solid #4CAF50;
            padding: 10px;
            font-size: 1.2em;
            color: #2E7D32;
        }
        .result-error {
            background-color: #FFEBEE;
            border-left: 5px solid #D32F2F;
            padding: 10px;
            font-size: 1.2em;
            color: #C62828;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the web app
st.markdown("<h1 class='main-title'>Fake News Detection App</h1>", unsafe_allow_html=True)

# Sub-header
st.markdown("<p class='sub-header'>Determine whether a news article is Fake or Real with AI</p>", unsafe_allow_html=True)

# Instructions for the user
st.write("### Enter the news article text below to check its validity")

# Text input box for news article
user_input = st.text_area("News Article:")

# Load pre-trained model and vectorizer
def load_model():
    try:
        model = joblib.load('fake_news_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer file not found. Please ensure 'fake_news_model.joblib' and 'tfidf_vectorizer.joblib' are in the same directory.")
        return None, None

# Load the model and vectorizer
model, vectorizer = load_model()

# Predict function
def predict(news):
    if model and vectorizer:
        # Preprocess and vectorize the input
        transformed_input = vectorizer.transform([news])
        prediction = model.predict(transformed_input)
        return prediction[0]
    else:
        return None

# Action button for prediction
if st.button("Check if it's Fake News"):
    if user_input.strip():
        with st.spinner("Analyzing the article..."):
            time.sleep(2)  # Simulate loading time
            prediction = predict(user_input)
            if prediction is not None:
                if prediction == 1:
                    st.markdown("<div class='result-error'>\U0001F6AB The news article is Fake.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-success'>\U0001F44D The news article is Real.</div>", unsafe_allow_html=True)
            else:
                st.error("Unable to make a prediction. Ensure the model is loaded correctly.")
    else:
        st.warning("Please enter some text in the input box.")

# Footer section
st.markdown("""
    ---
    **About**: This app uses a Machine Learning model to detect fake news based on text input. Created for demonstration purposes.
    
    *Developer*: Abdul Rafay
    """, unsafe_allow_html=True)
