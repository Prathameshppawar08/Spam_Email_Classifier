import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
import ssl

# Handle SSL certificate verification for environments with SSL issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Download NLTK datasets for punkt and stopwords if they are not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Ensure the punkt and stopwords are downloaded
try:
    stopwords.words('english')
    word_tokenize('test')
except LookupError:
    st.error("Required NLTK resources not found. Please try again later.")
    exit()  # Exit if resources are missing

ps = PorterStemmer()

# Preprocess text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = word_tokenize(text)  # Tokenize the text

    y = []
    for i in text:
        if i.isalnum():  # Remove non-alphanumeric characters
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess input
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize the input message using the loaded TF-IDF vectorizer
    vector_input = tfidf.transform([transformed_sms])

    # 3. Make prediction using the loaded model
    result = model.predict(vector_input)[0]

    # 4. Display prediction result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
