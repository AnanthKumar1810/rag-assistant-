from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def create_vectorstore(chunks):
    print("Embedding chunks and storing in ChromaDB...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    return vectorstore