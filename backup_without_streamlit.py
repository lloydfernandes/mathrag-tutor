# Local RAG Math Tutor using Ollama + ChromaDB + GSM8K
import os
import chromadb
from chromadb.utils import embedding_functions
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from glob import glob

# Step 1: Relpace with GSM8K Data - For now its just basic test data
def load_math_dataset(path="./data/gsm8k"):
    docs = []
    for file_path in glob(f"{path}/*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            docs.append(Document(page_content=content))
    return docs

# Step 2: Split documents
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = load_math_dataset()
split_docs = text_splitter.split_documents(documents)

# Step 3: Set up embedding function (local-compatible)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create or load Chroma vector store
chroma_path = "./chroma_math_db"
if not os.path.exists(chroma_path):
    vectordb = Chroma.from_documents(split_docs, embedding_model, persist_directory=chroma_path)
    vectordb.persist()
else:
    vectordb = Chroma(persist_directory=chroma_path, embedding_function=embedding_model)

# Step 5: Set up RAG chain with Ollama 
llm = Ollama(model="mistral")  # Should we change to Qwen?
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), return_source_documents=True)

# Step 6: Test with math question
while True:
    query = input("\nAsk a math question (or 'exit'): ")
    if query.lower() == 'exit':
        break
    result = rag_chain({"query": query})
    print("\nAnswer:\n", result['result'])
    print("\nRetrieved Context:\n")
    for doc in result['source_documents']:
        print("-", doc.page_content[:300], "...\n")


#----------- Backup

import streamlit as st
import litellm
import random

st.set_page_config(page_title="Chatbot - Powered by Open Source LLM")

## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    #message_placeholder = st.empty()
    full_response = ""
    #messages.append({"role": "user", "content": message.content})
    output =  litellm.completion(
            model="ollama/llama3.2",
            messages=prompt,
            api_base="http://localhost:11434",
            stream=True
    )
    #
    for chunk in output:
        if chunk:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
         #
    return full_response



st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by Ollama & Open Source LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = generate_response(st.session_state.messages)
    msg = response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)