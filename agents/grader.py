class GraderAgent:
    def grade(self, question: str, student_answer: str, correct_answer: str) -> str:
        if student_answer.strip() == correct_answer.strip():
            return "✅ Correct! Great job."
        else:
            return f"❌ Incorrect. Correct answer was: {correct_answer}"