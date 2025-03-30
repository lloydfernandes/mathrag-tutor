class SessionTracker:
    def __init__(self):
        self.attempts = []
        self.mistakes = []

    def log_attempt(self, question: str, student_answer: str, correct_answer: str, feedback: str):
        record = {
            "question": question,
            "student_answer": student_answer,
            "correct_answer": correct_answer,
            "feedback": feedback
        }
        self.attempts.append(record)

        if "✅" not in feedback:
            self.mistakes.append(record)

    def get_history(self):
        return self.attempts

    def get_mistakes(self):
        return self.mistakes

    def get_score(self):
        correct = sum(1 for attempt in self.attempts if "✅" in attempt["feedback"])
        total = len(self.attempts)
        return correct, total