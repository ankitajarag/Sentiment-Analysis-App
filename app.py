{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1f2eaed-9949-4d57-894e-61eb0c7e1c75",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import re\n",
    "import string\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "# Load model and vectorizer\n",
    "model = pickle.load(open(\"sentiment_model.pkl\", \"rb\"))\n",
    "vectorizer = pickle.load(open(\"tfidf_vectorizer.pkl\", \"rb\"))\n",
    "\n",
    "# Stopwords\n",
    "nltk.download('stopwords')\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "# Text cleaning function\n",
    "def clean_text(text):\n",
    "    text = text.lower()\n",
    "    text = re.sub(r\"http\\S+\", \"\", text)\n",
    "    text = re.sub(r\"\\d+\", \"\", text)\n",
    "    text = text.translate(str.maketrans(\"\", \"\", string.punctuation))\n",
    "    words = text.split()\n",
    "    words = [w for w in words if w not in stop_words]\n",
    "    return \" \".join(words)\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"Product Review Sentiment Analysis\")\n",
    "\n",
    "st.write(\"Enter a review and the model will predict sentiment.\")\n",
    "\n",
    "user_input = st.text_area(\"Enter Review\")\n",
    "\n",
    "if st.button(\"Predict Sentiment\"):\n",
    "\n",
    "    clean_review = clean_text(user_input)\n",
    "\n",
    "    vector = vectorizer.transform([clean_review])\n",
    "\n",
    "    prediction = model.predict(vector)\n",
    "\n",
    "    st.subheader(\"Prediction:\")\n",
    "    st.success(prediction[0])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
