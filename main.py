# 📁 mathrag/main.py

import streamlit as st
from agents.explainer import ExplainerAgent
from agents.grader import GraderAgent
from agents.supervisor import SupervisorAgent
from agents.memory_agent import MemoryAgent
from practice.practice import PracticeAgent
from practice.session_tracker import SessionTracker

from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import json
from glob import glob
import os
import random

st.set_page_config(page_title="Intelligent Math Tutor")
st.title("📚 Intelligent Math Tutor")

# Dataset toggle
dataset_option = st.sidebar.selectbox("Choose dataset:", ["GSM8K Full", "Test Set"])

@st.cache_resource
def load_docs(dataset: str):
    docs = []
    if dataset == "Test Set":
        for file_path in glob("./data/gsm8k_test/*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                docs.append(Document(page_content=content))
    else:
        path = "./data/gsm8k/train.jsonl"
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                q = item.get("question", "").strip()
                a = item.get("answer", "").strip()
                if q and a:
                    docs.append(Document(page_content=f"Question: {q}\n\nAnswer: {a}"))
    return docs

documents = load_docs(dataset_option)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_path = f"./chroma_math_db_{'test' if dataset_option == 'Test Set' else 'full'}"
if not os.path.exists(chroma_path):
    vectordb = Chroma.from_documents(split_docs, embedding_model, persist_directory=chroma_path)
    vectordb.persist()
else:
    vectordb = Chroma(persist_directory=chroma_path, embedding_function=embedding_model)

custom_prompt = PromptTemplate.from_template("""
You are a helpful and obedient math tutor. You must ONLY use the provided context to answer the question.
If the answer is not in the context, say "I don't know."
<context>
{context}
</context>

Question: {question}
Answer:
""")

# Initialize agents
if ("supervisor" not in st.session_state or
    "practice_agent" not in st.session_state or
    st.session_state.active_dataset != dataset_option):

    llm = Ollama(model="mistral")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )

    explainer = ExplainerAgent(rag_chain, llm)
    grader = GraderAgent()
    supervisor = SupervisorAgent(explainer, grader)
    practice_agent = PracticeAgent(documents, grader)
    tracker = SessionTracker()
    memory = MemoryAgent(tracker)

    st.session_state.supervisor = supervisor
    st.session_state.practice_agent = practice_agent
    st.session_state.tracker = tracker
    st.session_state.memory = memory
    st.session_state.explainer = explainer
    st.session_state.active_dataset = dataset_option

supervisor = st.session_state.supervisor
practice_agent = st.session_state.practice_agent
tracker = st.session_state.tracker
memory = st.session_state.memory
explainer = st.session_state.explainer

mode = st.radio("Select mode:", ["Explain", "Grade", "Practice", "Progress", "Session Log"])

if mode == "Explain":
    question = st.text_input("Enter a math question:")
    if question:
        answer, docs = supervisor.handle_query("explain", question=question)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Context")
        for i, doc in enumerate(docs):
            st.markdown(f"**Doc {i+1}:** {doc.page_content[:300]}...")

elif mode == "Grade":
    question = st.text_input("Question:")
    correct_answer = st.text_input("Correct Answer:")
    student_answer = st.text_input("Student's Answer:")

    if question and correct_answer and student_answer:
        feedback = supervisor.handle_query(
            "grade",
            question=question,
            correct_answer=correct_answer,
            student_answer=student_answer
        )
        st.subheader("Feedback")
        st.write(feedback)

elif mode == "Practice":
    st.subheader("🎯 Practice Mode")

    st.markdown("**Question Source:**")
    mistake_mode = st.toggle("Practice mistakes only")

    if "current_question" not in st.session_state:
        st.session_state.current_question = None

    if st.button("🌀 New Question") or not st.session_state.current_question:
        if mistake_mode and tracker.get_mistakes():
            st.session_state.current_question = random.choice(tracker.get_mistakes())
        else:
            q_obj = practice_agent.get_random_question()
            st.session_state.current_question = q_obj
        st.session_state.practice_feedback = ""

    current_q = st.session_state.current_question
    st.write(f"**Question:** {current_q['question']}")
    student_input = st.text_input("Your Answer:")

    if st.button("Submit Answer") and student_input:
        feedback, correct = practice_agent.grade_answer(current_q['question'], student_input)
        topic = explainer.classify_topic(current_q['question'])
        tracker.log_attempt(current_q['question'], student_input, correct, feedback, topic)
        st.session_state.practice_feedback = feedback

    if st.session_state.get("practice_feedback"):
        st.subheader("Feedback")
        st.write(st.session_state.practice_feedback)

    correct, total = tracker.get_score()
    st.caption(f"📊 Score: {correct}/{total} correct")

elif mode == "Progress":
    st.subheader("📈 Session Progress")
    stats = memory.get_session_stats()
    st.write(stats)

    topic_stats = tracker.get_topic_stats()
    if topic_stats:
        st.markdown("### 📚 Topic Breakdown")
        for topic, result in topic_stats.items():
            accuracy = (result['correct'] / result['total']) * 100 if result['total'] else 0
            st.markdown(f"- **{topic.title()}**: {result['correct']} / {result['total']} correct ({accuracy:.1f}%)")

    st.markdown("### 📤 Export Session Log")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export as JSON"):
            path = memory.export_json()
            st.success(f"✅ Saved to {path}")
    with col2:
        if st.button("Export as CSV"):
            path = memory.export_csv()
            st.success(f"✅ Saved to {path}")

elif mode == "Session Log":
    st.subheader("📝 Session Activity Log")
    logs = supervisor.get_session_log() + tracker.get_history()
    if not logs:
        st.info("No activity yet. Use 'Explain', 'Grade' or 'Practice' to populate the log.")
    for entry in logs:
        st.markdown("---")
        st.json(entry)
