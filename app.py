"""
Simple web interface for the Fake News Detector.
Run:  streamlit run app.py
"""

import os

import joblib
import streamlit as st

from train import MODEL_PATH, clean_text

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Paste a news headline or article below and the ML model will predict if it is **Real** or **Fake**.")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


saved = load_model()
st.caption(f"Model: {saved['name']} · Test accuracy: {saved['accuracy'] * 100:.1f}%")

news = st.text_area("News text", height=200)

if st.button("Check News", type="primary"):
    if not news.strip():
        st.warning("Please enter some text.")
    else:
        model = saved["model"]
        cleaned = clean_text(news)
        label = model.predict([cleaned])[0]
        if label == 1:
            st.error("🚨 This news looks **FAKE**")
        else:
            st.success("✅ This news looks **REAL**")

        # Show confidence when the model supports probabilities
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([cleaned])[0][label]
            st.progress(float(prob), text=f"Confidence: {prob * 100:.1f}%")

with st.expander("Model evaluation plots"):
    for img in ["model_comparison.png", "confusion_matrix.png", "label_distribution.png"]:
        path = os.path.join("outputs", img)
        if os.path.exists(path):
            st.image(path)
