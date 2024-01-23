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

# Function for Text Summarization App
def text_summarization_app():
    st.title("Text Summarizer")
    
    # Sidebar for navigation
    st.sidebar.header("Choose App")
    app_choice = st.sidebar.radio("", ["Text Summarizer", "Sentiment Analyzer"])

    if app_choice == "Text Summarizer":
        st.subheader("Analyze Text")
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

    elif app_choice == "Sentiment Analyzer":
        st.subheader("Analyze Document (PDF)")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        if uploaded_file is not None:
            with st.spinner("Analyzing document..."):
                with st.container():
                    st.info("Document uploaded successfully")
                    extracted_text = extract_text_from_pdf(uploaded_file.name)
                    st.markdown("**Extracted Text is Below:**")
                    st.info(extracted_text)

            if st.button("Analyze Text"):
                sentiment_result = analyze_sentiment(extracted_text)
                st.write(f"Sentiment: {sentiment_result}")

# Function for Sentiment Analysis App
def sentiment_analysis_app():
    st.title("Sentiment Analyzer")
    
    # Sidebar for navigation
    st.sidebar.header("Choose App")
    app_choice = st.sidebar.radio("", ["Text Summarizer", "Sentiment Analyzer"])

    if app_choice == "Text Summarizer":
        st.subheader("Analyze Text")
        input_text = st.text_area("Enter your text here")

        if st.button("Analyze Sentiment"):
            if input_text:
                sentiment_result = analyze_sentiment(input_text)
                st.write(f"Sentiment: {sentiment_result}")
            else:
                st.warning("Please enter some text for analysis.")

    elif app_choice == "Sentiment Analyzer":
        st.subheader("Analyze CSV Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Display the uploaded data
            st.subheader("Uploaded Data:")
            st.write(df)

            # Analyze sentiment for each row in the uploaded data
            df['Sentiment'] = df['Text'].apply(analyze_sentiment)

            # Display sentiment analysis results
            st.subheader("Sentiment Analysis Results:")
            st.write(df[['Text', 'Sentiment']])

            # EDA Section
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

    st.title("Text and Sentiment Analysis App")

    if st.sidebar.button("Reset"):
        st.caching.clear_cache()

    if st.sidebar.checkbox("Show Code", False):
        st.sidebar.code(
            """
            # Your code goes here
            """
        )

    if st.sidebar.checkbox("Hide Sidebar", False):
        st.sidebar.button("Show Sidebar")

    st.sidebar.header("Choose App")
    app_choice = st.sidebar.radio("", ["Text Summarizer", "Sentiment Analyzer"])

    if app_choice == "Text Summarizer":
        text_summarization_app()
    elif app_choice == "Sentiment Analyzer":
        sentiment_analysis_app()

if __name__ == "__main__":
    main()
