import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from PyPDF2 import PdfReader
from textblob import TextBlob

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to perform text summarization
@st.cache(suppress_st_warning=True)
def text_summary(text, maxlength=None):
    summarizer = pipeline("summarization")
    result = summarizer(text, max_length=150)  # Adjust max_length as needed
    return result[0]['summary_text']

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

# Function for Sentiment Analysis App
def sentiment_analysis_app():
    st.title("Sentiment Analysis App")
    
    # User's choice: Analyze text or CSV dataset
    analysis_choice = st.radio("Choose analysis option:", ["Analyze Text", "Analyze Document"])

    if analysis_choice == "Analyze Text":
        # Text area for sentiment analysis
        st.subheader("Enter text for sentiment analysis:")
        user_input = st.text_area("Input text here:")

        if st.button("Analyze Sentiment"):
            if user_input:
                sentiment_result = analyze_sentiment(user_input)
                st.write(f"Sentiment: {sentiment_result}")
            else:
                st.warning("Please enter some text for analysis.")

    elif analysis_choice == "Analyze Document":
        # File uploader for Document
        st.subheader("Upload a document (PDF) for sentiment analysis:")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded_file is not None:
            with st.spinner("Analyzing document..."):
                with st.container():
                    st.info("Document uploaded successfully")
                    extracted_text = extract_text_from_pdf(uploaded_file.name)
                    st.markdown("**Extracted Text is Below:**")
                    st.info(extracted_text)

            if st.button("Analyze Sentiment"):
                sentiment_result = analyze_sentiment(extracted_text)
                st.write(f"Sentiment: {sentiment_result}")

# Function for Text Summarization App
def text_summarization_app():
    st.title("Text Summarization App")
    
    # Text area for text summarization
    st.subheader("Enter text for summarization:")
    input_text = st.text_area("Enter your text here")

    if st.button("Summarize Text"):
        if input_text:
            with st.spinner("Summarizing text..."):
                with st.container():
                    st.markdown("**Your Input Text**")
                    st.info(input_text)
                    result = text_summary(input_text)
                    st.markdown("**Summary Result**")
                    st.success(result)

# Function to perform EDA for sentiment analysis results
def perform_eda(df):
    st.subheader("Exploratory Data Analysis (EDA):")
    
    # Bar chart for sentiment distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(plt)

    # Display average sentiment score
    avg_sentiment = df['Sentiment'].value_counts(normalize=True).idxmax()
    st.write(f"Overall Sentiment: {avg_sentiment}")

# Combined App
def main():
    st.set_page_config(layout="wide")

    st.title("Combined Text and Sentiment Analysis App")

    # Sidebar for navigation
    app_choice = st.sidebar.radio("Select App", ["Text Summarization", "Sentiment Analysis"])

    if app_choice == "Text Summarization":
        text_summarization_app()
    elif app_choice == "Sentiment Analysis":
        sentiment_analysis_app()
        df = pd.DataFrame()  # Placeholder for sentiment analysis results
        perform_eda(df)

if __name__ == "__main__":
    main()
