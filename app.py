import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# Create a summarization pipeline using a pre-trained model
summarizer = pipeline("summarization")
# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

@st.cache_resource
def text_summary(text, maxlength=None):
    result = summarizer(text, max_length=150)  # Adjust max_length as needed
    return result[0]['summary_text']

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

# Streamlit app title
st.title("Text and Sentiment Analyzer")

# Sidebar for selecting the analysis type
analysis_type = st.sidebar.radio("Select Analysis Type:", ["Text Summarization", "Sentiment Analysis"])

if analysis_type == "Text Summarization":
    st.subheader("Text Summarization using Hugging Face Transformers")
    input_text = st.text_area("Enter your text here")
    if input_text is not None and st.button("Summarize"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Your Input Text**")
            st.info(input_text)
        with col2:
            st.markdown("**Summary Result**")
            result = text_summary(input_text)
            st.success(result)

elif analysis_type == "Sentiment Analysis":
    st.subheader("Sentiment Analysis using Hugging Face Transformers")
    analysis_input = st.radio("Select Analysis Input:", ["Single Text", "CSV File"])

    if analysis_input == "Single Text":
        # Input for single text
        single_text = st.text_area("Enter a single text for analysis:")
        if st.button("Analyze"):
            if single_text:
                result = sentiment_analyzer(single_text)
                st.write(f"Sentiment: {result[0]['label']} with confidence: {result[0]['score']:.4f}")

    elif analysis_input == "CSV File":
        # Input for CSV file
        csv_file = st.file_uploader("Upload a CSV file for batch analysis:", type=["csv"])

        if csv_file:
            # Read CSV file into DataFrame with 'latin-1' encoding
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except UnicodeDecodeError:
                st.warning("Error decoding file with utf-8 encoding. Trying lati
