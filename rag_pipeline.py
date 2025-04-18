import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def carregar_banco_vetorial(caminho="chroma_atas"):
    embedding_engine = HuggingFaceEmbeddings(model_name="distilbert-base-nli-stsb-mean-tokens")
    return Chroma(persist_directory=caminho, embedding_function=embedding_engine)

def responder_pergunta(pergunta, vector_db, api_key):
    retriever = vector_db.as_retriever(k=3)
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")
    prompt = hub.pull("rlm/rag-prompt")
    
    rag = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag.invoke(pergunta)
