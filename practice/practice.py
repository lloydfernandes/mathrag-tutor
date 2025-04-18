import random
import re
from langchain.schema import Document
from agents.grader import GraderAgent

class PracticeAgent:
    def __init__(self, documents: list[Document], grader: GraderAgent):
        self.questions = self._extract_questions(documents)
        self.grader = grader

    def _extract_questions(self, docs):
        qna_pairs = []
        for doc in docs:
            content = doc.page_content
            if "Question:" in content and "Answer:" in content:
                try:
                    q = content.split("Question:")[1].split("Answer:")[0].strip()
                    a = content.split("Answer:")[1].strip()
                    qna_pairs.append({"question": q, "answer": a})
                except:
                    continue
        return qna_pairs

    def _clean_answer(self, answer: str) -> str:
        cleaned = re.sub(r"<<[^<>]+>>", "", answer)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def get_random_question(self):
        return random.choice(self.questions)

    def grade_answer(self, question, student_answer):
        match = next((q for q in self.questions if q["question"] == question), None)
        if not match:
            return "⚠️ Question not found.", ""

        correct = self._clean_answer(match["answer"])
        print("💬 Student Answer:", student_answer)
        print("✅ Correct Answer:", correct)

        feedback = self.grader.grade(question, student_answer, correct)
        return feedback, correct
