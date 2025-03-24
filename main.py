# Streamlit Math RAG Tutor (local-only)

import os   
import streamlit as st
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
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Local Math Tutor", page_icon="📚")
st.title("📚 Local Math Tutor (Offline RAG)")
st.markdown("Ask a math question. It will retrieve examples from your local dataset and generate a helpful answer using a local LLM (via Ollama). 🧠")

# Step 1: Load GSM8K Data (or your own dataset)
@st.cache_resource
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

# Step 3: Set up embedding function
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create/load Chroma vector store
chroma_path = "./chroma_math_db"
if not os.path.exists(chroma_path):
    vectordb = Chroma.from_documents(split_docs, embedding_model, persist_directory=chroma_path)
    vectordb.persist()
else:
    vectordb = Chroma(persist_directory=chroma_path, embedding_function=embedding_model)

# Step 5: Set up RAG chain with custom prompt to use context only
custom_prompt = PromptTemplate.from_template("""
You are a helpful and obedient math tutor. You must ONLY use the provided context to answer the question.
If the answer is not in the context, say \"I don't know.\"

<context>
{context}
</context>

Question: {question}
Answer:
""")

llm = Ollama(model="mistral")
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

# Step 6: Streamlit input form
query = st.text_input("Ask a math question:", placeholder="e.g., A boy had 5 candies and ate 2. How many are left?")

if query:
    result = rag_chain.invoke({"query": query})
    st.subheader("🧠 Answer")
    st.write(result['result'])

    st.subheader("📄 Retrieved Context")
    for i, doc in enumerate(result['source_documents']):
        st.markdown(f"**Example {i+1}:**\n\n{doc.page_content[:500]}...")