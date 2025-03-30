from agents.explainer import ExplainerAgent
from agents.grader import GraderAgent

class SupervisorAgent:
    def __init__(self, explainer: ExplainerAgent, grader: GraderAgent):
        self.explainer = explainer
        self.grader = grader
        self.session_log = []

    def handle_query(self, mode: str, **kwargs):
        if mode == "explain":
            answer, docs = self.explainer.explain(kwargs.get("question"))
            self.session_log.append({
                "mode": "explain",
                "question": kwargs.get("question"),
                "answer": answer,
                "context": [doc.page_content[:200] for doc in docs]
            })
            return answer, docs

        elif mode == "grade":
            feedback = self.grader.grade(
                kwargs.get("question"),
                kwargs.get("student_answer"),
                kwargs.get("correct_answer")
            )
            self.session_log.append({
                "mode": "grade",
                "question": kwargs.get("question"),
                "student_answer": kwargs.get("student_answer"),
                "correct_answer": kwargs.get("correct_answer"),
                "feedback": feedback
            })
            return feedback

        else:
            return "Unknown mode."

    def get_session_log(self):
        return self.session_log