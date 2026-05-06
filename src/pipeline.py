from dspy_modules import (
    QueryGenerator,
    AnswerGenerator,
    CorruptedAnswerGenerator,
    AnswerRepairer,
    IterativeQueryRepairer,
)


class ForwardRepairPipeline:
    def __init__(self, retriever, corrupt_stage: str = "query"):
        if corrupt_stage not in ("query", "answer"):
            raise ValueError(f"corrupt_stage must be 'query' or 'answer', got {corrupt_stage!r}")

        self.retriever = retriever
        self.corrupt_stage = corrupt_stage

        self.baseline_query_generator = QueryGenerator(mode="baseline")
        self.answer_generator = AnswerGenerator()

        if corrupt_stage == "query":
            self.corrupted_query_generator = QueryGenerator(mode="corrupted")
            self.repaired_query_generator = QueryGenerator(mode="repaired")
            self.iterative_query_repairer = IterativeQueryRepairer()
        else:
            self.corrupted_answer_generator = CorruptedAnswerGenerator()
            self.answer_repairer = AnswerRepairer()

    def _retrieve(self, query: str) -> tuple[list[dict], str]:
        docs = self.retriever.retrieve(query)
        context = "\n".join(
            f"Title: {doc['title']}\nText: {doc['text']}" for doc in docs
        )
        return docs, context

    def run_baseline(self, question: str) -> dict:
        query = self.baseline_query_generator(question=question).query
        docs, context = self._retrieve(query)
        answer = self.answer_generator(question=question, context=context).answer
        return {"query": query, "docs": docs, "context": context, "answer": answer}

    def run_corrupted(self, question: str) -> dict:
        if self.corrupt_stage == "query":
            query = self.corrupted_query_generator(question=question).query
            docs, context = self._retrieve(query)
            answer = self.answer_generator(question=question, context=context).answer
        else:
            query = self.baseline_query_generator(question=question).query
            docs, context = self._retrieve(query)
            answer = self.corrupted_answer_generator(question=question, context=context).answer
        return {"query": query, "docs": docs, "context": context, "answer": answer}

    def run_repaired(
        self,
        question: str,
        bad_query: str | None = None,
        bad_answer: str | None = None,
    ) -> dict:
        if self.corrupt_stage == "query":
            if bad_query is None:
                raise ValueError("bad_query required for corrupt_stage='query'")
            query = self.repaired_query_generator(question=question, bad_query=bad_query).query
            docs, context = self._retrieve(query)
            answer = self.answer_generator(question=question, context=context).answer
        else:
            if bad_answer is None:
                raise ValueError("bad_answer required for corrupt_stage='answer'")
            query = self.baseline_query_generator(question=question).query
            docs, context = self._retrieve(query)
            answer = self.answer_repairer(
                question=question, context=context, bad_answer=bad_answer
            ).answer
        return {"query": query, "docs": docs, "context": context, "answer": answer}

    def run_repaired_iterative(self, question: str, bad_query: str) -> dict:
        result = self.iterative_query_repairer(question=question, bad_query=bad_query)
        docs = self.retriever.retrieve_union(
            [result.query_a, result.query_b],
            top_k_total=self.retriever.top_k,
        )
        context = "\n".join(
            f"Title: {doc['title']}\nText: {doc['text']}" for doc in docs
        )
        answer = self.answer_generator(question=question, context=context).answer
        return {
            "reasoning": result.reasoning,
            "query_a": result.query_a,
            "query_b": result.query_b,
            "docs": docs,
            "context": context,
            "answer": answer,
        }
