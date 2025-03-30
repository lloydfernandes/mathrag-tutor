import re

class GraderAgent:
    def grade(self, question: str, student_answer: str, correct_answer: str) -> str:
        clean_student = self._normalize(student_answer)
        clean_correct = self._normalize(correct_answer)

        # Try numeric match if both are numbers
        try:
            if float(clean_student) == float(clean_correct):
                return "✅ Correct!"
        except:
            pass

        # Extract final number from correct_answer (e.g., #### 30)
        final_number_match = re.search(r"####\s*(\d+(\.\d+)?)", correct_answer)
        if final_number_match:
            final_number = final_number_match.group(1)
            try:
                if float(clean_student) == float(final_number):
                    return "✅ Correct!"
            except:
                pass

        # Partial string match
        if clean_student in clean_correct or clean_correct in clean_student:
            return "✅ Correct!"

        return f"❌ Incorrect. Correct answer was: {correct_answer}"

    def _normalize(self, text):
        text = text.lower().strip()
        text = text.replace("hours", "").replace("hour", "").replace("minutes", "").replace("minute", "")
        text = text.replace("=", "").replace(":", "").replace(",", "")
        return text.strip()