from time import perf_counter

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

    @staticmethod
    def _telemetry(
        started_at: float,
        query_generation: float,
        retrieval: float,
        answer_generation: float,
    ) -> dict:
        return {
            "wall_clock_seconds": perf_counter() - started_at,
            "latency_seconds": {
                "query_generation": query_generation,
                "retrieval": retrieval,
                "answer_generation": answer_generation,
            },
        }

    def run_baseline(self, question: str) -> dict:
        started_at = perf_counter()
        stage_start = perf_counter()
        query = self.baseline_query_generator(question=question).query
        query_latency = perf_counter() - stage_start
        stage_start = perf_counter()
        docs, context = self._retrieve(query)
        retrieval_latency = perf_counter() - stage_start
        stage_start = perf_counter()
        answer = self.answer_generator(question=question, context=context).answer
        answer_latency = perf_counter() - stage_start
        return {
            "query": query,
            "docs": docs,
            "context": context,
            "answer": answer,
            "telemetry": self._telemetry(
                started_at, query_latency, retrieval_latency, answer_latency
            ),
        }

    def run_corrupted(self, question: str) -> dict:
        started_at = perf_counter()
        stage_start = perf_counter()
        if self.corrupt_stage == "query":
            query = self.corrupted_query_generator(question=question).query
            query_latency = perf_counter() - stage_start
            stage_start = perf_counter()
            docs, context = self._retrieve(query)
            retrieval_latency = perf_counter() - stage_start
            stage_start = perf_counter()
            answer = self.answer_generator(question=question, context=context).answer
        else:
            query = self.baseline_query_generator(question=question).query
            query_latency = perf_counter() - stage_start
            stage_start = perf_counter()
            docs, context = self._retrieve(query)
            retrieval_latency = perf_counter() - stage_start
            stage_start = perf_counter()
            answer = self.corrupted_answer_generator(question=question, context=context).answer
        answer_latency = perf_counter() - stage_start
        return {
            "query": query,
            "docs": docs,
            "context": context,
            "answer": answer,
            "telemetry": self._telemetry(
                started_at, query_latency, retrieval_latency, answer_latency
            ),
        }

    def run_repaired(
        self,
        question: str,
        bad_query: str | None = None,
        bad_answer: str | None = None,
    ) -> dict:
        started_at = perf_counter()
        stage_start = perf_counter()
        if self.corrupt_stage == "query":
            if bad_query is None:
                raise ValueError("bad_query required for corrupt_stage='query'")
            query = self.repaired_query_generator(question=question, bad_query=bad_query).query
            query_latency = perf_counter() - stage_start
            stage_start = perf_counter()
            docs, context = self._retrieve(query)
            retrieval_latency = perf_counter() - stage_start
            stage_start = perf_counter()
            answer = self.answer_generator(question=question, context=context).answer
        else:
            if bad_answer is None:
                raise ValueError("bad_answer required for corrupt_stage='answer'")
            query = self.baseline_query_generator(question=question).query
            query_latency = perf_counter() - stage_start
            stage_start = perf_counter()
            docs, context = self._retrieve(query)
            retrieval_latency = perf_counter() - stage_start
            stage_start = perf_counter()
            answer = self.answer_repairer(
                question=question, context=context, bad_answer=bad_answer
            ).answer
        answer_latency = perf_counter() - stage_start
        return {
            "query": query,
            "docs": docs,
            "context": context,
            "answer": answer,
            "telemetry": self._telemetry(
                started_at, query_latency, retrieval_latency, answer_latency
            ),
        }

    def run_repaired_iterative(self, question: str, bad_query: str) -> dict:
        started_at = perf_counter()
        stage_start = perf_counter()
        result = self.iterative_query_repairer(question=question, bad_query=bad_query)
        query_latency = perf_counter() - stage_start
        stage_start = perf_counter()
        docs = self.retriever.retrieve_union(
            [result.query_a, result.query_b],
            top_k_total=self.retriever.top_k,
        )
        context = "\n".join(
            f"Title: {doc['title']}\nText: {doc['text']}" for doc in docs
        )
        retrieval_latency = perf_counter() - stage_start
        stage_start = perf_counter()
        answer = self.answer_generator(question=question, context=context).answer
        answer_latency = perf_counter() - stage_start
        return {
            "reasoning": result.reasoning,
            "query_a": result.query_a,
            "query_b": result.query_b,
            "docs": docs,
            "context": context,
            "answer": answer,
            "telemetry": self._telemetry(
                started_at, query_latency, retrieval_latency, answer_latency
            ),
        }
