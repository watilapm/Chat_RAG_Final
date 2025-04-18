import streamlit as st
import os
from rag_pipeline import carregar_banco_vetorial, responder_pergunta

st.title("🧠 Chat com RAG Ibama")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = carregar_banco_vetorial()

openai_api_key = st.secrets["OPENAI_API_KEY"]

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Faça sua pergunta:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = responder_pergunta(prompt, st.session_state.vector_db, openai_api_key)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
