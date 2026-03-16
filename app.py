import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load model
model = pickle.load(open("sentiment_model.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

st.title("Sentiment Analysis App")

user_input = st.text_area("Enter Review")

if st.button("Predict"):

    cleaned = clean_text(user_input)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    st.write("Sentiment:", prediction[0])
