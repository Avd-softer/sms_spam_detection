import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --------------------------
# Ensure required NLTK resources are available
# --------------------------
nltk_resources = ["punkt", "stopwords"]
for resource in nltk_resources:
    try:
        if resource == "punkt":
            nltk.data.find("tokenizers/punkt")
        else:
            nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

# --------------------------
# Initialize stemmer
# --------------------------
ps = PorterStemmer()

# --------------------------
# Text preprocessing function
# --------------------------
def transform_text(text):
    # Lowercase
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Remove special characters (keep only alphanumeric)
    text = [word for word in text if word.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = [word for word in text if word not in stop_words]

    # Stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# --------------------------
# Load trained model & vectorizer
# --------------------------
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# --------------------------
# Streamlit UI
# --------------------------
st.title("📧 Email/SMS Spam Classifier")

input_text = st.text_area("Enter your text")

if st.button("Predict"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_text)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)

        if result[0] == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")
