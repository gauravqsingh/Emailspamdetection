import streamlit as st
from src.predict import predict_email  # adjust if function name is different

st.title("📧 Email Spam Detection App")

# Input from user
email_text = st.text_area("Enter the email text here:")

if st.button("Predict"):
    try:
        prediction = predict_email(email_text)   # call your function
        if prediction == 1:
            st.error("🚨 This email is **SPAM**")
        else:
            st.success("✅ This email is **NOT Spam**")
    except Exception as e:
        st.write("Error while predicting:", e)
