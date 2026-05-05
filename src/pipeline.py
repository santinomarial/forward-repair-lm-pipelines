from dspy_modules import QueryGenerator, AnswerGenerator


class ForwardRepairPipeline:
    def __init__(self, retriever):
        self.retriever = retriever

        self.baseline_query_generator = QueryGenerator(mode="baseline")
        self.corrupted_query_generator = QueryGenerator(mode="corrupted")
        self.repaired_query_generator = QueryGenerator(mode="repaired")

        self.answer_generator = AnswerGenerator()

    def _answer_from_query(self, question: str, query: str) -> dict:
        docs = self.retriever.retrieve(query)
        context = "\n".join(
            f"Title: {doc['title']}\nText: {doc['text']}" for doc in docs
        )

        answer = self.answer_generator(
            question=question,
            context=context,
        ).answer

        return {
            "query": query,
            "docs": docs,
            "context": context,
            "answer": answer,
        }

    def run_baseline(self, question: str) -> dict:
        query = self.baseline_query_generator(question=question).query
        return self._answer_from_query(question, query)

    def run_corrupted(self, question: str) -> dict:
        query = self.corrupted_query_generator(question=question).query
        return self._answer_from_query(question, query)

    def run_repaired(self, question: str, bad_query: str) -> dict:
        query = self.repaired_query_generator(
            question=question,
            bad_query=bad_query,
        ).query
        return self._answer_from_query(question, query)