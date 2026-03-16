import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords safely
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# Load model
model = pickle.load(open("sentiment_model.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("Product Review Sentiment Analysis")

review = st.text_area("Enter a review")

if st.button("Predict Sentiment"):
    cleaned_review = clean_text(review)
    vector = vectorizer.transform([cleaned_review])
    prediction = model.predict(vector)

    st.success(f"Predicted Sentiment: {prediction[0]}")
