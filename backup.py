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

# Step 1: Load GSM8K Data (or your own dataset)
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

# Step 5: Set up RAG chain with Ollama (Mistral/Qwen/etc.)
llm = Ollama(model="mistral")  # Change to "qwen2" if desired
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), return_source_documents=True)

# Step 6: Ask a math question
while True:
    query = input("\nAsk a math question (or 'exit'): ")
    if query.lower() == 'exit':
        break
    result = rag_chain({"query": query})
    print("\nAnswer:\n", result['result'])
    print("\nRetrieved Context:\n")
    for doc in result['source_documents']:
        print("-", doc.page_content[:300], "...\n")
