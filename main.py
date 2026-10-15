from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from rag_utils import load_pdf, split_documents, create_retriever, ask_question

app = FastAPI(title="Document Q&A System")

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

retriever_store = {}


class ChatRequest(BaseModel):
    question: str
    groq_api_key: str
    model_name: str = "llama-3.3-70b-versatile"


class ChatResponse(BaseModel):
    answer: str
    source_pages: list


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    documents = load_pdf(file_path)
    if not documents:
        raise HTTPException(status_code=400, detail="Could not extract any text from the PDF.")

    chunks = split_documents(documents)
    retriever = create_retriever(chunks)
    retriever_store["current"] = retriever

    return {
        "message": f"'{file.filename}' uploaded and processed successfully.",
        "total_chunks": len(chunks)
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if "current" not in retriever_store:
        raise HTTPException(status_code=400, detail="No document uploaded yet. Please upload a PDF first.")

    try:
        result = ask_question(
            retriever_store["current"],
            request.question,
            request.groq_api_key,
            request.model_name
        )
        return ChatResponse(answer=result["answer"], source_pages=result["source_pages"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
