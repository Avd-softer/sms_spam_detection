import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Use local nltk_data folder
nltk.data.path.append('./nltk_data')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    stop_words = set(stopwords.words("english"))
    text = [word for word in text if word not in stop_words]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Email/SMS Spam Classifier")
input_text = st.text_area("Enter your text")

if st.button("Predict"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        transformed_sms = transform_text(input_text)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)
        st.header("🚨 Spam" if result[0] == 1 else "✅ Not Spam")
