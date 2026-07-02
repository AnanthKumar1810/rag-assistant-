import os
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()  # loads API key from .env file

PROMPT_TEMPLATE = """You are a research paper assistant.
Answer ONLY using the context below.
Cite page numbers where possible.
If the answer is not in the context, say "Not found in document."

Context:
{context}

Question: {question}
Answer:"""


def build_chain(vectorstore):
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain