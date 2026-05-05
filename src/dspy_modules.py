import dspy


class GenerateQuery(dspy.Signature):
    """Generate a precise retrieval query for answering the question."""
    question = dspy.InputField()
    query = dspy.OutputField(desc="A precise search query with named entities and key relationships.")


class GenerateCorruptedQuery(dspy.Signature):
    """Generate a vague, underspecified retrieval query that omits useful details."""
    question = dspy.InputField()
    query = dspy.OutputField(desc="A vague search query that omits named entities, dates, or relationships.")


class RepairQuery(dspy.Signature):
    """Repair a vague retrieval query so it preserves the information needed to answer the question."""
    question = dspy.InputField()
    bad_query = dspy.InputField()
    query = dspy.OutputField(desc="A repaired precise search query.")


class GenerateAnswer(dspy.Signature):
    """Answer the question using only the retrieved context. If the context does not contain enough evidence, answer UNKNOWN."""
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField(
        desc="The final answer only. Use UNKNOWN if the retrieved context is insufficient."
    )


class QueryGenerator(dspy.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

        if mode == "baseline":
            self.generate = dspy.Predict(GenerateQuery)
        elif mode == "corrupted":
            self.generate = dspy.Predict(GenerateCorruptedQuery)
        elif mode == "repaired":
            self.generate = dspy.Predict(RepairQuery)
        else:
            raise ValueError(f"Unknown query generation mode: {mode}")

    def forward(self, question: str, bad_query: str | None = None):
        if self.mode == "repaired":
            if bad_query is None:
                raise ValueError("bad_query is required for repaired mode")
            return self.generate(question=question, bad_query=bad_query)

        return self.generate(question=question)


class AnswerGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateAnswer)

    def forward(self, question: str, context: str):
        guarded_context = (
            "Rules:\n"
            "1. Use only the retrieved context below.\n"
            "2. Do not use outside knowledge.\n"
            "3. If the context does not explicitly support the answer, output UNKNOWN.\n\n"
            f"{context}"
        )
        return self.generate(question=question, context=guarded_context)