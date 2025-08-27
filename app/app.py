import streamlit as st
from pathlib import Path
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'spam_classifier.pkl'

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run training first (python -m src.train)")
        return None
    return load(MODEL_PATH)

st.set_page_config(page_title="Email/SMS Spam Detector", page_icon="📧")

st.title("📧 Email / SMS Spam Detector")
st.write("Paste any message or email text below. The model will predict whether it's **Spam** or **Ham (Not Spam)**.")

model = load_model()

user_text = st.text_area("Enter message:", height=180, placeholder="e.g., Congratulations! You won a lottery. Call now...")

if st.button("Predict"):
    if not model:
        st.stop()
    if not user_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        pred = model.predict([user_text])[0]
        prob = model.predict_proba([user_text]).max()
        if pred == 'spam':
            st.error(f"Prediction: **SPAM**  |  Confidence: {prob:.2%}")
        else:
            st.success(f"Prediction: **HAM (Not Spam)**  |  Confidence: {prob:.2%}")

st.caption("Model: TF-IDF + Multinomial Naive Bayes. Trained on your dataset in /data/spam.csv")