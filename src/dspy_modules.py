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


class GenerateCorruptedAnswer(dspy.Signature):
    """Answer the question using only your own general knowledge. Do NOT use or refer to the retrieved context in any way."""
    question = dspy.InputField()
    context = dspy.InputField(desc="[IGNORE] Retrieved passages — do not use these.")
    answer = dspy.OutputField(desc="Your answer based solely on general knowledge, not on the context above.")


class RepairAnswer(dspy.Signature):
    """The previous answer ignored the retrieved evidence. Produce a corrected answer grounded strictly in the retrieved context."""
    question = dspy.InputField()
    context = dspy.InputField()
    bad_answer = dspy.InputField(desc="The previous ungrounded answer — do not trust it.")
    answer = dspy.OutputField(
        desc="Corrected answer grounded in the retrieved context. Use UNKNOWN if the context is insufficient."
    )


class GenerateIterativeRepairQuery(dspy.Signature):
    """Given a vague retrieval query and the original question, identify the two distinct entities or sub-questions that must be retrieved to answer it.
    First reason explicitly about what the bad query is missing, then produce two targeted queries — one per entity or sub-question."""
    question = dspy.InputField()
    bad_query = dspy.InputField(desc="The vague, underspecified query that failed to retrieve the right documents.")
    reasoning = dspy.OutputField(desc="Brief analysis of what named entities or sub-questions the bad query misses.")
    query_a = dspy.OutputField(desc="Query targeting the primary entity or first sub-question.")
    query_b = dspy.OutputField(desc="Query targeting the secondary or missing entity or second sub-question.")


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


class IterativeQueryRepairer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateIterativeRepairQuery)

    def forward(self, question: str, bad_query: str):
        return self.generate(question=question, bad_query=bad_query)


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


class CorruptedAnswerGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateCorruptedAnswer)

    def forward(self, question: str, context: str):
        return self.generate(question=question, context=context)


class AnswerRepairer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(RepairAnswer)

    def forward(self, question: str, context: str, bad_answer: str):
        guarded_context = (
            "Rules:\n"
            "1. Use only the retrieved context below.\n"
            "2. Do not use outside knowledge.\n"
            "3. If the context does not explicitly support the answer, output UNKNOWN.\n\n"
            f"{context}"
        )
        return self.generate(question=question, context=guarded_context, bad_answer=bad_answer)