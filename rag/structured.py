"""Schema-validated answers with explicit source and abstention behavior."""
from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="Answer based only on supplied passages; empty if absent")
    source_pages: list[int] = Field(description="Zero-based page indices supporting the answer")
    evidence_quotes: list[str] = Field(description="Exact supporting substrings copied from retrieved passages")
    found: bool = Field(description="False when passages do not contain the answer")


def _ask_once(vectorstore, question: str, llm, k: int):
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(f"[page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs)
    prompt = ("Answer only using these PDF passages. If they do not establish the answer, "
              "set found=false, answer='', source_pages=[] and evidence_quotes=[]. "
              "Quote exact supporting spans and cite zero-based page indices.\n"
              f"Passages:\n{context}\n\nQuestion: {question}")
    structured = llm.with_structured_output(Answer, include_raw=True)
    result = structured.invoke(prompt)
    if result.get("parsing_error") or not isinstance(result.get("parsed"), Answer):
        # One bounded schema retry; malformed markdown/trailing text never becomes a tool result.
        result = structured.invoke(prompt + "\nYour previous response failed schema validation. Return only the required structured fields.")
        if result.get("parsing_error") or not isinstance(result.get("parsed"), Answer):
            raise ValueError("Model response failed the structured answer schema after one retry")
    answer = result["parsed"]
    allowed = {d.metadata.get("page") for d in docs}
    if (answer.found and (not answer.answer.strip() or not answer.source_pages or not answer.evidence_quotes)) or (
        not answer.found and (answer.answer.strip() or answer.source_pages or answer.evidence_quotes)
    ) or any(page not in allowed for page in answer.source_pages) or any(
        not quote.strip() or not any(quote in d.page_content and d.metadata.get("page") in answer.source_pages
                                     for d in docs) for quote in answer.evidence_quotes):
        raise ValueError("Structured answer references absent evidence or contradicts found status")
    return answer, docs


def ask_structured(vectorstore, question: str, llm):
    """Use top-3; retry once with one extra passage only after explicit abstention."""
    answer, docs = _ask_once(vectorstore, question, llm, 3)
    if not answer.found:
        return _ask_once(vectorstore, question, llm, 4)
    return answer, docs

