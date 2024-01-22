import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# Create sentiment analysis pipeline using a pre-trained model
sentiment_analyzer = pipeline("sentiment-analysis")

# Create summarization pipeline using a pre-trained model
summarizer = pipeline("summarization")

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
st.title("Text and Sentiment Analysis App")

# Sidebar for selecting analysis type
analysis_type = st.sidebar.radio("Select Analysis Type:", ["Sentiment Analysis", "Text Summarization"])

if analysis_type == "Sentiment Analysis":
    # Sentiment Analysis Section
    st.subheader("Sentiment Analysis using Hugging Face Transformers")

    # Choose analysis type: Single Text or CSV File
    analysis_subtype = st.sidebar.radio("Select Analysis Subtype:", ["Single Text", "CSV File"])

    if analysis_subtype == "Single Text":
        # Input for single text
        single_text = st.text_area("Enter a single text for analysis:")

        if st.button("Analyze"):
            if single_text:
                result = sentiment_analyzer(single_text)
                st.write(f"Sentiment: {result[0]['label']} with confidence: {result[0]['score']:.4f}")

    elif analysis_subtype == "CSV File":
        # Input for CSV file
        csv_file = st.file_uploader("Upload a CSV file for batch analysis:", type=["csv"])

        if csv_file:
            # Read CSV file into DataFrame with 'latin-1' encoding
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except UnicodeDecodeError:
                st.warning("Error decoding file with utf-8 encoding. Trying latin-1 encoding.")
                df = pd.read_csv(csv_file, encoding='latin-1')

            # Check if any column contains 'text'
            text_column = next((col for col in df.columns if 'text' in col.lower()), None)

            if text_column is None:
                st.error("CSV file must contain a column containing the word 'text'")
            else:
                # Analyze sentiments for each text in the identified column
                sentiments = sentiment_analyzer(df[text_column].tolist())

                # Add new columns for sentiment and confidence in the DataFrame
                df["sentiment"] = [res["label"] for res in sentiments]
                df["confidence"] = [res["score"] for res in sentiments]

                # Display the results
                st.dataframe(df)

                # EDA Section
                st.subheader("Exploratory Data Analysis (EDA)")

                # Display distribution of sentiments
                st.write("Sentiment Distribution:")
                sns.countplot(x="sentiment", data=df)
                st.pyplot()

                # Display confidence distribution
                st.write("Confidence Distribution:")
                plt.hist(df["confidence"], bins=20, color='skyblue', edgecolor='black')
                st.pyplot()

elif analysis_type == "Text Summarization":
    # Text Summarization Section
    st.subheader("Text Summarization using Hugging Face Transformers")

    # User choice
    choice = st.sidebar.radio("Select Summarization Choice:", ["Analyze Single Text", "Analyze Document"])

    if choice == "Analyze Single Text":
        # Input for single text
        input_text = st.text_area("Enter your text here")
        if input_text is not None and st.button("Analyze"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Analysis Result**")
                result = text_summary(input_text)
                st.success(result)

    elif choice == "Analyze Document":
        # Input for PDF document
        input_file = st.file_uploader("Upload your document here", type=['pdf'])
        if input_file is not None and st.button("Analyze"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1, 1])
            with col1:
                st.info("File uploaded successfully")
                extracted_text = extract_text_from_pdf("doc_file.pdf")
                st.markdown("**Extracted Text is Below:**")
                st.info(extracted_text)
            with col2:
                st.markdown("**Analysis Result**")
                doc_summary = text_summary(extracted_text)
                st.success(doc_summary)
