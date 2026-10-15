import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Document Q&A System", page_icon="📄", layout="wide")
st.title("Document Q&A System")


MODELS = [
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
]


with st.sidebar:
    st.header(" Settings")
    groq_api_key = st.text_input("Groq API Key", type="password")

    model_name = st.selectbox("Groq Model", MODELS, index=0)

    st.divider()
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file and st.button(" Process Document"):
        with st.spinner("Uploading and processing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"{data['message']} ({data['total_chunks']} chunks created)")
                else:
                    st.error(f"{response.json().get('detail', 'Upload failed')}")
            except requests.ConnectionError:
                st.error("Cannot connect to the backend. Make sure FastAPI server is running.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your document..."):
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question,
                        "groq_api_key": groq_api_key,
                        "model_name": model_name
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    pages = ", ".join(str(p) for p in data["source_pages"])
                    full_response = f"{data['answer']}\n\n---\n**Source Pages:** {pages}"
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    error_msg = response.json().get("detail", "Something went wrong")
                    st.error(f"{error_msg}")
            except requests.ConnectionError:
                st.error("Cannot connect to the backend. Make sure FastAPI server is running.")
