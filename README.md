# 📚 Local RAG Math Tutor (Offline Chatbot)

A fully offline, local chatbot that answers math questions by retrieving relevant examples from a custom knowledge base using **RAG (Retrieval-Augmented Generation)** and a **locally running LLM via Ollama**.

---

## 🚀 Features

- 🧠 **RAG-powered**: Retrieves real math examples to ground its answers.
- 🧩 **Uses local LLMs**: No internet, no OpenAI API needed.
- 🧾 **Your own knowledge base**: Add math problems in `.txt` files.
- ⚡️ **Fast + Private**: Everything runs locally using ChromaDB and Ollama.
- 💬 **Streamlit interface**: Simple web app for interaction.

---

## 📁 Project Structure

```
mathrag/
├── main.py                 # Streamlit app
├── requirements.txt        # Python dependencies
├── .gitignore              # Files/folders to exclude from Git
├── data/
│   └── gsm8k/              # Your math problems (.txt files)
│       ├── problem1.txt
│       ├── problem2.txt
├── chroma_math_db/         # Auto-generated vector DB (ignored by Git)
```

---

## 🧠 How It Works

1. Loads math problem files from `data/gsm8k/`
2. Splits and embeds them using `sentence-transformers`
3. Stores embeddings in a local Chroma vector store
4. Uses retrieval to fetch similar examples for any user query
5. Feeds query + context into a local LLM like `mistral` via **Ollama**
6. Answers with contextually grounded reasoning

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone git@github.com:lloydfernandes/mathrag-tutor.git
cd mathrag-tutor
```

### 2. Set Up a Conda Environment

```bash
conda create -n mathrag python=3.10
conda activate mathrag
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install & Start Ollama

Download Ollama from [https://ollama.com](https://ollama.com)

Then run a local model:
```bash
ollama run mistral
# or
ollama run qwen:7b
```

### 5. Add Math Problems

Put your `.txt` math problems in:
```
data/gsm8k/
```

**Example (`problem1.txt`)**
```
Question: Jane has 3 apples. She buys 4 more. How many apples does she have now?
Answer: 3 + 4 = 7. Jane has 7 apples.
```

### 6. Run the App

```bash
streamlit run main.py
```

Open the app in your browser: `http://localhost:8501`

---

## 💡 Customization Tips

- 📄 Add more `.txt` files to make your tutor smarter.
- 🧪 Use made-up facts (like “Mars capital is Cheesetown”) to test RAG.
- 🔒 Toggle “strict mode” (coming soon) to force the LLM to use only the context.

---

## 🧪 Example Questions to Try

```
A boy had 5 candies. He ate 2. How many are left?
What is the capital of Mars?  <-- test custom data
A train leaves at 3PM and arrives at 7PM. How long was the journey?
```

---

## 📦 Dependencies

```
langchain
chromadb
sentence-transformers
transformers
torch
streamlit
```

(You can install these via `pip install -r requirements.txt`)

---

## 🧠 Credits & Acknowledgments

- Built on top of [LangChain](https://github.com/langchain-ai/langchain)
- Uses [ChromaDB](https://github.com/chroma-core/chroma) for local vector search
- Powered by open-source LLMs (via [Ollama](https://ollama.com))

---

## 🔐 License

MIT License — free to use, modify, and share!