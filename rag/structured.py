"""Schema-validated answers with explicit source and abstention behavior."""
from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="Answer based only on supplied passages; empty if absent")
    source_pages: list[int] = Field(description="Zero-based page indices supporting the answer")
    found: bool = Field(description="False when passages do not contain the answer")


def ask_structured(vectorstore, question: str, llm):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(f"[page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs)
    prompt = ("Answer only using these PDF passages. If they do not establish the answer, "
              "set found=false, answer='' and source_pages=[]. Cite zero-based page indices.\n"
              f"Passages:\n{context}\n\nQuestion: {question}")
    result = llm.with_structured_output(Answer, include_raw=True).invoke(prompt)
    if result.get("parsing_error") or not isinstance(result.get("parsed"), Answer):
        raise ValueError("Model response failed the structured answer schema")
    answer = result["parsed"]
    allowed = {d.metadata.get("page") for d in docs}
    if (answer.found and (not answer.answer.strip() or not answer.source_pages)) or (
        not answer.found and (answer.answer.strip() or answer.source_pages)
    ) or any(page not in allowed for page in answer.source_pages):
        raise ValueError("Structured answer references absent evidence or contradicts found status")
    return answer, docs

