import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources (safe to call every time, won't redownload if already present)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Lowercase, tokenization, removing special characters, removing stopwords, stemming
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


# Load trained model & vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit UI
st.title("📧 Email/SMS Spam Classifier")

input_text = st.text_area("Enter your text")

if st.button("Predict"):
    # preprocess
    transformed_sms = transform_text(input_text)

    # vectorize
    vector_input = tfidf.transform([transformed_sms])

    # predict
    result = model.predict(vector_input)

    if result == 1:
        st.header("🚨 Spam")
    else:
        st.header("✅ Not Spam")
