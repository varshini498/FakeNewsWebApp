import streamlit as st
from src.predictor import predict_news

# Page configuration
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# Title
st.markdown("<h1 style='text-align:center;'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered web app to detect misinformation</p>", unsafe_allow_html=True)
st.divider()

# Input
news_text = st.text_area("Paste the news article below:", height=200, placeholder="Enter news here...")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Predict button
if st.button("Analyze News"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, confidence, keywords = predict_news(news_text)
        result = "REAL NEWS ✅" if label == 1 else "FAKE NEWS ❌"

        st.subheader("Prediction Result")
        if label == 1:
            st.success(result)
        else:
            st.error(result)

        st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")

        st.subheader("Top Influential Keywords")
        for word, score in keywords:
            st.write(f"- **{word}**")

        st.session_state.history.append(result)

st.divider()

# Show session history
if st.session_state.history:
    st.subheader("Prediction History (Session)")
    for i, res in enumerate(st.session_state.history, 1):
        st.write(f"{i}. {res}")
