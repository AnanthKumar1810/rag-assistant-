"""PDF ingestion, overlapping passages, MiniLM embeddings and Chroma retrieval."""
from dataclasses import dataclass
import hashlib

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    page_content: str
    metadata: dict


_model = None


def get_embeddings():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def load_and_chunk(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks = []
    for page_number, page in enumerate(PdfReader(pdf_path).pages):
        text = page.extract_text() or ""
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundaries = [text.rfind(sep, start + chunk_size // 2, end)
                              for sep in ("\n\n", "\n", ". ", " ")]
                end = next((point + 1 for point in boundaries if point > start), end)
            content = text[start:end].strip()
            if content:
                chunks.append(Document(content, {"page": page_number, "source": pdf_path}))
            if end == len(text):
                break
            start = max(start + 1, end - chunk_overlap)
    if not chunks:
        raise ValueError("The PDF has no extractable text")
    return chunks


class VectorStore:
    def __init__(self, collection):
        self.collection = collection

    def similarity_search(self, question: str, k: int = 3):
        query = get_embeddings().encode([question], normalize_embeddings=True).tolist()
        found = self.collection.query(query_embeddings=query, n_results=k,
                                      include=["documents", "metadatas"])
        return [Document(text, meta) for text, meta in
                zip(found["documents"][0], found["metadatas"][0])]


def create_vectorstore(chunks, persist_directory: str | None = "./chroma_db"):
    if not chunks:
        raise ValueError("At least one passage is required")
    settings = Settings(anonymized_telemetry=False)
    client = (chromadb.PersistentClient(path=persist_directory, settings=settings) if persist_directory
              else chromadb.EphemeralClient(settings=settings))
    digest = hashlib.sha256("\n".join(
        f"{doc.metadata.get('page')}:{doc.page_content}" for doc in chunks
    ).encode("utf-8")).hexdigest()[:24]
    collection = client.get_or_create_collection("paper_" + digest,
                                                  metadata={"hnsw:space": "cosine"})
    vectors = get_embeddings().encode([doc.page_content for doc in chunks],
                                      normalize_embeddings=True).tolist()
    collection.upsert(ids=[str(i) for i in range(len(chunks))],
                      documents=[doc.page_content for doc in chunks],
                      metadatas=[{"page": doc.metadata["page"],
                                  "source": str(doc.metadata.get("source", ""))} for doc in chunks],
                      embeddings=vectors)
    return VectorStore(collection)

