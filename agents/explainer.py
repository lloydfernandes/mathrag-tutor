class ExplainerAgent:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

    def explain(self, question: str) -> str:
        result = self.rag_chain.invoke({"query": question})
        return result['result'], result['source_documents']