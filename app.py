import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


# Load saved model
model = pickle.load(open("sentiment_model.pkl", "rb"))

# Load TFIDF vectorizer
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

st.title("Sentiment Analysis App")

review = st.text_area("Enter a review")

if st.button("Predict Sentiment"):
    cleaned_review = clean_text(review)
    vector = vectorizer.transform([cleaned_review])
    prediction = model.predict(vector)

    st.success(f"Predicted Sentiment: {prediction[0]}")
