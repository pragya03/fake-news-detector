import streamlit as st
import pickle
import os
import csv
import requests
import urllib.parse
import datetime
from source_checker import check_source_reliability, get_page_title, google_search_news

def load_custom_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css("style.css")

API_KEY = st.secrets["API_KEY"]
CSE_ID = st.secrets["CSE_ID"]

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("Fake News Detector with Source Verification")

log_file = "logs.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "News Text", "Prediction", "URL"])

# UI with columns
st.subheader("Enter News Article")

col1, col2 = st.columns(2)

with col1:
    news_text = st.text_area("News Content", height=200, key="news_input")
with col2:
    url_input = st.text_input("News URL (Optional)", key="url_input")

# --- Main Prediction Section ---
if st.button("Analyze News", key="analyze_button"):
    if news_text.strip() == "":
        st.warning("Please enter the news content.")
    else:
        # Vectorization and ML Prediction
        input_vec = vectorizer.transform([news_text])
        prediction = model.predict(input_vec)[0]
        prediction_label = "Fake News" if prediction == 0 else "Real News"
        st.success(f"Model Prediction: {'🟥 Fake News' if prediction == 0 else '🟩 Real News'}")

        # Default values for logging
        source_status = "N/A"
        page_title = ""
        top_link = ""

        # Step 1: Google Real-Time Search
        st.info("Searching for matching news online...")
        results = google_search_news(news_text, API_KEY, CSE_ID)

        if results and "Error" not in results[0][0]:
            st.success("Found the following similar news articles:")
            for title, link in results:
                st.markdown(f"- [{title}]({link})")
            top_link = results[0][1]
        if prediction == 0 and results:
            st.warning("Model says it's Fake, but similar articles exist online.")
            final_verdict = "Possibly Real"
        elif prediction == 1 and not results:
            st.warning("Model says it's Real, but no trusted articles found online.")
            final_verdict = "Unverified"
        elif prediction == 1 and results:
            st.success("Real News — both model and internet agree.")
            final_verdict = "Likely Real"
        else:
            st.error("Likely Fake — no similar articles and model agrees.")
            final_verdict = "Likely Fake"

        # Step 2: If user gave a URL, check source reliability
        if url_input.strip():
            st.subheader("Verifying Provided Source")
            source_status = check_source_reliability(url_input)
            page_title = get_page_title(url_input)

            st.info(f"Source Status: {source_status}")
            st.write("**News Title:**", page_title)
            st.markdown(f"[Open Article in New Tab]({url_input})", unsafe_allow_html=True)

        prob = model.predict_proba(input_vec)[0]
        confidence = max(prob) * 100
        st.caption(f"Model confidence: {confidence:.2f}%")

        # Step 3: Log Everything to CSV
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, news_text, prediction_label, url_input if url_input else top_link])

        st.success("Logged successfully to `logs.csv`")
