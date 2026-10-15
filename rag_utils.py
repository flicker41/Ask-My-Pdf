import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def load_pdf(file_path):
    pdf = pymupdf.open(file_path)
    documents = []
    for page_num in range(len(pdf)):
        text = pdf[page_num].get_text()
        if text.strip():
            documents.append(Document(page_content=text, metadata={"page": page_num + 1}))
    pdf.close()
    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)


def create_retriever(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_store = FAISS.from_documents(chunks, embeddings)

    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    return {"faiss": faiss_store, "bm25": bm25, "chunks": chunks}


def hybrid_search(retriever_data, query, k=5):
    faiss_store = retriever_data["faiss"]
    bm25 = retriever_data["bm25"]
    chunks = retriever_data["chunks"]

    faiss_docs = faiss_store.similarity_search(query, k=k)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs = [chunks[i] for i in top_bm25_indices]

    seen = set()
    merged = []
    for doc in faiss_docs + bm25_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)

    return merged[:k]


def ask_question(retriever_data, question, api_key, model_name="llama-3.3-70b-versatile"):
    top_docs = hybrid_search(retriever_data, question)

    context_parts = []
    source_pages = set()
    for doc in top_docs:
        page = doc.metadata.get("page", "?")
        source_pages.add(page)
        context_parts.append(f"[Page {page}]\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    llm = ChatGroq(
        model_name=model_name,
        groq_api_key=api_key,
        temperature=0
    )

    system_prompt = (
        "You are a helpful document assistant. Answer the user's question strictly "
        "based on the provided context. Always mention the relevant source page numbers "
        "in your answer. If the answer is not found in the context, say "
        "'I could not find the answer in the provided document.'"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "source_pages": sorted(source_pages)
    }
