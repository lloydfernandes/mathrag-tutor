# 📚 Local RAG Math Tutor (Offline Chatbot)

A fully offline, local chatbot that answers math questions by retrieving relevant examples from a custom knowledge base using **RAG (Retrieval-Augmented Generation)** and a **locally running LLM via Ollama**.

---

## 🚀 Features

- 🧠 **RAG-powered**: Retrieves real math examples to ground its answers.
- 🧩 **Uses local LLMs**: No internet, no OpenAI API needed.
- 🧾 **Your own knowledge base**: Add math problems in `.txt` or `.jsonl` files.
- ⚡️ **Fast + Private**: Everything runs locally using ChromaDB and Ollama.
- 💬 **Streamlit interface**: Simple web app for interaction.
- 📊 **Progress tracking**: See your accuracy, topic breakdown, and export logs.
- 🧠 **LLM-based topic tagging**: Automatically classifies each question into a math topic.

---

## 📁 Project Structure

```
mathrag/
├── main.py                 # Streamlit app
├── requirements.txt        # Python dependencies
├── .gitignore              # Files/folders to exclude from Git
├── agents/                 # Core agents
│   ├── explainer.py        # ExplainerAgent with RAG and topic classifier
│   ├── grader.py           # GraderAgent for checking answers
│   ├── supervisor.py       # Routes queries to agents
│   └── memory_agent.py     # Tracks and exports performance data
├── practice/
│   ├── practice.py         # PracticeAgent for question serving
│   └── session_tracker.py  # Tracks attempts per session
├── data/
│   └── gsm8k/              # Math dataset (.jsonl or .txt)
├── chroma_math_db*/        # Auto-generated vector DB (ignored by Git)
```

---

## 🧠 How It Works

1. Loads math problem files from `data/gsm8k/`
2. Splits and embeds them using `sentence-transformers`
3. Stores embeddings in a local Chroma vector store
4. Uses retrieval to fetch similar examples for any user query
5. Feeds query + context into a local LLM like `mistral` via **Ollama**
6. Classifies topic using LLM for every practice question
7. Tracks and exports accuracy and performance stats

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

Put your `.txt` or `train.jsonl` math problems in:
```
data/gsm8k/
```

**Example (`problem1.txt`)**
```
Question: Jane has 3 apples. She buys 4 more. How many apples does she have now?
Answer: 3 + 4 = 7. Jane has 7 apples.
```

**OR** use the official GSM8K dataset in `train.jsonl` format.

### 6. Run the App

```bash
streamlit run main.py
```

Open the app in your browser: `http://localhost:8501`

---

## 🧪 Modes to Explore

- 🧠 **Explain**: Ask any math question and get a grounded answer
- 🧪 **Practice**: Answer random or weak-topic questions
- 🎯 **Grade**: Compare your answer to a correct one
- 📈 **Progress**: View accuracy and performance by topic
- 📝 **Session Log**: Review every question and feedback

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

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🧠 Credits & Acknowledgments

- Built on top of [LangChain](https://github.com/langchain-ai/langchain)
- Uses [ChromaDB](https://github.com/chroma-core/chroma) for local vector search
- Powered by open-source LLMs (via [Ollama](https://ollama.com))
- Dataset based on [GSM8K](https://huggingface.co/datasets/gsm8k)

---

## 🔐 License

MIT License — free to use, modify, and share!
