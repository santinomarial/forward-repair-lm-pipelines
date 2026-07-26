from time import perf_counter

from dspy_modules import (
    QueryGenerator,
    AnswerGenerator,
    CorruptedAnswerGenerator,
    AnswerRepairer,
    IterativeQueryRepairer,
)
from routing import FailureDetector, RepairAction, RepairPolicy


class ForwardRepairPipeline:
    def __init__(self, retriever, corrupt_stage: str = "query"):
        if corrupt_stage not in ("query", "answer"):
            raise ValueError(f"corrupt_stage must be 'query' or 'answer', got {corrupt_stage!r}")

        self.retriever = retriever
        self.corrupt_stage = corrupt_stage

        self.baseline_query_generator = QueryGenerator(mode="baseline")
        self.answer_generator = AnswerGenerator()
        self.corrupted_query_generator = QueryGenerator(mode="corrupted")
        self.repaired_query_generator = QueryGenerator(mode="repaired")
        self.iterative_query_repairer = IterativeQueryRepairer()
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
        if self.corrupt_stage == "query":
            if bad_query is None:
                raise ValueError("bad_query required for corrupt_stage='query'")
            return self.run_query_repair(question=question, bad_query=bad_query)

        if bad_answer is None:
            raise ValueError("bad_answer required for corrupt_stage='answer'")
        return self.run_answer_repair(question=question, bad_answer=bad_answer)

    def run_query_repair(self, question: str, bad_query: str) -> dict:
        started_at = perf_counter()
        stage_start = perf_counter()
        query = self.repaired_query_generator(question=question, bad_query=bad_query).query
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

    def run_answer_repair(
        self,
        question: str,
        bad_answer: str,
        *,
        query: str | None = None,
        docs: list[dict] | None = None,
        context: str | None = None,
    ) -> dict:
        started_at = perf_counter()
        if query is None or docs is None or context is None:
            stage_start = perf_counter()
            query = self.baseline_query_generator(question=question).query
            query_latency = perf_counter() - stage_start
            stage_start = perf_counter()
            docs, context = self._retrieve(query)
            retrieval_latency = perf_counter() - stage_start
        else:
            query_latency = 0.0
            retrieval_latency = 0.0

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

    @staticmethod
    def _combined_telemetry(
        initial: dict,
        final: dict,
        *,
        wall_clock_seconds: float,
    ) -> dict:
        initial_latency = initial["telemetry"]["latency_seconds"]
        final_latency = final["telemetry"]["latency_seconds"]
        return {
            "wall_clock_seconds": wall_clock_seconds,
            "latency_seconds": {
                stage: float(initial_latency.get(stage, 0.0))
                + float(final_latency.get(stage, 0.0))
                for stage in (
                    "query_generation",
                    "retrieval",
                    "answer_generation",
                )
            },
        }

    def run_adaptive(
        self,
        question: str,
        *,
        detector: FailureDetector,
        policy: RepairPolicy,
        initial_run: dict | None = None,
    ) -> dict:
        """Diagnose a completed run and apply only the selected repair."""
        started_at = perf_counter()
        supplied_initial = initial_run is not None
        initial = initial_run if supplied_initial else self.run_baseline(question)

        decision_started_at = perf_counter()
        signals = detector.detect(
            question=question,
            query=initial["query"],
            docs=initial["docs"],
            answer=initial["answer"],
        )
        action = policy.decide(signals)
        decision_latency = perf_counter() - decision_started_at

        if action == RepairAction.ACCEPT:
            final = dict(initial)
        elif action == RepairAction.REPAIR_QUERY:
            final = self.run_query_repair(question, bad_query=initial["query"])
        elif action == RepairAction.REPAIR_ANSWER:
            final = self.run_answer_repair(
                question,
                bad_answer=initial["answer"],
                query=initial["query"],
                docs=initial["docs"],
                context=initial["context"],
            )
        elif action == RepairAction.REPAIR_ITERATIVE:
            final = self.run_repaired_iterative(question, bad_query=initial["query"])
        else:
            raise ValueError(f"Unsupported repair action: {action!r}")

        if supplied_initial:
            elapsed = (
                float(initial["telemetry"]["wall_clock_seconds"])
                + perf_counter()
                - started_at
            )
        else:
            elapsed = perf_counter() - started_at

        if action == RepairAction.ACCEPT:
            telemetry = {
                "wall_clock_seconds": elapsed,
                "latency_seconds": dict(initial["telemetry"]["latency_seconds"]),
            }
        else:
            telemetry = self._combined_telemetry(
                initial,
                final,
                wall_clock_seconds=elapsed,
            )

        final["telemetry"] = telemetry
        final["routing"] = {
            "policy": policy.name,
            "action": action.value,
            "repaired": action != RepairAction.ACCEPT,
            "decision_latency_seconds": decision_latency,
            "signals": signals.to_dict(),
        }
        final["initial"] = {
            "query": initial["query"],
            "doc_ids": [doc["id"] for doc in initial["docs"]],
            "answer": initial["answer"],
        }
        return final
