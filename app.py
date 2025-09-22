import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --------------------------
# Setup NLTK resources safely
# --------------------------
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.mkdir(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Download required resources if missing
for resource in ["punkt", "stopwords"]:
    try:
        if resource == "punkt":
            nltk.data.find("tokenizers/punkt")
        else:
            nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

ps = PorterStemmer()

# --------------------------
# Preprocessing function
# --------------------------
def transform_text(text):
    # Lowercase
    text = text.lower()
    # Tokenization
    text = nltk.word_tokenize(text)

    y = []
    # Remove special characters (only alphanumeric kept)
    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords & punctuation
    text = []
    stop_words = set(stopwords.words('english'))  # load once for speed
    for i in y:
        if i not in stop_words and i not in string.punctuation:
            text.append(i)

    # Stemming
    y = [ps.stem(i) for i in text]

    return " ".join(y)

# --------------------------
# Load model & vectorizer
# --------------------------
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# --------------------------
# Streamlit UI
# --------------------------
st.title("📧 Email/SMS Spam Classifier")

input_text = st.text_area("Enter your text")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_text)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)

        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")
