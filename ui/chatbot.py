import streamlit as st
import requests

st.title("GenAI Q&A Chatbot")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    try:
        response = requests.post(
            "http://localhost:3750/ask",
            json={"question": question}
        )
        if response.status_code == 200:
            st.success(f"Answer: {response.json()['answer']}")
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
