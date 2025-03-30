import json
import csv
from datetime import datetime

class MemoryAgent:
    def __init__(self, tracker):
        self.tracker = tracker

    def get_session_stats(self):
        attempts = self.tracker.get_history()
        correct, total = self.tracker.get_score()
        accuracy = round((correct / total) * 100, 2) if total > 0 else 0

        return {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy_percent": accuracy
        }

    def export_json(self, filename=None):
        filename = filename or f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.tracker.get_history(), f, indent=2)
        return filename

    def export_csv(self, filename=None):
        filename = filename or f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["question", "student_answer", "correct_answer", "feedback"])
            writer.writeheader()
            writer.writerows(self.tracker.get_history())
        return filename