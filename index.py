import numpy as np
import streamlit as st
import spacy
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout

def preprocess_text(text):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text.lower())
        tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
        return ' '.join(tokens)

class LLM:    
    def predict_sentiment(self, user_input):

        # Load the saved tokenizer and model
        with open('src/models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        model = load_model('src/models/trained_models/lstm_model.h5')

        # Preprocess the user input
        cleaned_text = preprocess_text(user_input)
        
        # Convert the input text into sequences and pad it
        input_seq = tokenizer.texts_to_sequences([cleaned_text])
        input_pad = pad_sequences(input_seq, maxlen=100)
        
        # Predict using the trained model
        prediction = model.predict(input_pad)
        
        # Convert the prediction to the class label
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Mapping back to original sentiment
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        return sentiment_map[predicted_class]

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis App", page_icon=":speech_balloon:", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f2f6fc;
        font-family: Arial, sans-serif;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1 {
        text-align: center;
        color: #2e86de;
        margin-bottom: 30px;
    }
    .textarea-box {
        border: 2px solid #2e86de;
        border-radius: 8px;
        padding: 10px;
        font-size: 18px;
        margin-bottom: 20px;
        width: 100%;
    }
    .submit-button {
        background-color: #2e86de;
        border: none;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 18px;
        cursor: pointer;
    }
    .result-box {
        font-size: 24px;
        padding: 20px;
        border: 2px solid #dfe6e9;
        border-radius: 8px;
        margin-top: 20px;
        text-align: center;
        color: #2d3436;
        background-color: #f1f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Title of the page
st.markdown("<h1>Sentiment Analysis App</h1>", unsafe_allow_html=True)

# Text Area for review input
user_input = st.text_area("Enter your review here:", height=150, key="textarea", max_chars=500, help="Max 500 characters")

# Submit button
if st.button("Analyze Sentiment", key="submit_button"):
    if user_input:

        llm_model = LLM()
        sentiment = llm_model.predict_sentiment(user_input)
        
        # Analyze and display the result
        if sentiment == 'positive':
            st.markdown('<div class="result-box" style="color: green;">Positive Sentiment</div>', unsafe_allow_html=True)
        elif sentiment == 'negative':
            st.markdown('<div class="result-box" style="color: red;">Negative Sentiment</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box" style="color: gray;">Neutral Sentiment</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text for analysis.")