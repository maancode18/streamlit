import streamlit as st
from io import StringIO
from PyPDF2 import PdfReader
import pandas as pd

# Import your summarizer logic directly
from model import SmartSummarizerPro

st.set_page_config(page_title="Smart Summarizer Pro", layout="centered")
st.title("🧠 Smart Summarizer Pro")
st.markdown("Summarize a URL, upload a PDF/TXT/CSV file, or paste raw text.")

# Initialize summarizer once
summarizer = SmartSummarizerPro()

# Choose input method
input_mode = st.radio("Choose input type:", ["URL or raw text", "Upload file (PDF, TXT, or CSV)"])

source_input = ""
if input_mode == "URL or raw text":
    source_input = st.text_area("Enter URL or raw text:", height=200)
else:
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "csv"])
    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                source_input = text.strip()

            elif uploaded_file.type == "text/plain":
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                source_input = stringio.read()

            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                source_input = df.astype(str).apply(" ".join, axis=1).str.cat(sep=" ")

            else:
                st.error("Unsupported file type.")

        except Exception as e:
            st.error(f"Failed to process file: {e}")

if st.button("Summarize"):
    if not source_input.strip():
        st.warning("Please provide valid input or upload a file.")
    else:
        try:
            summary = summarizer.run({"source": source_input})
            st.subheader("📝 Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"Summarization failed:\n\n{e}")
