class ExplainerAgent:
    def __init__(self, rag_chain, llm):
        self.rag_chain = rag_chain
        self.llm = llm

    def explain(self, question: str):
        response = self.rag_chain({"query": question})
        answer = response.get("result", "No answer returned.")
        docs = response.get("source_documents", [])
        return answer, docs

    def classify_topic(self, question: str) -> str:
        prompt = (
            "Classify the following math question into the single most appropriate topic category:"
            " addition, subtraction, multiplication, division, unit conversion,"
            " algebra, time & date, geometry, or other."
            "\n\nRespond ONLY with one of these topic names."
            f"\n\nQuestion: {question}\n\nTopic:"
        )
        try:
            result_raw = self.llm(prompt)
            result = result_raw.strip().lower()
            print("\n======================")
            print("🔎 QUESTION:", question)
            print("🔎 RAW LLM RESPONSE:", result_raw)
            print("======================\n")

            valid_topics = [
                "addition", "subtraction", "multiplication", "division",
                "unit conversion", "algebra", "time & date", "geometry", "other"
            ]

            if result in valid_topics:
                return result
            else:
                return "other"
        except Exception as e:
            print("[classify_topic error]", e)
            return "unknown"