import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = joblib.load("final_sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("Flipkart Review Sentiment Analyzer")

review = st.text_area("Enter Product Review")

if st.button("Predict Sentiment"):

    cleaned = clean_text(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😡")
