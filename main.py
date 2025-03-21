from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str

app = FastAPI()

# Environment validation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

# Document processing
def initialize_qa_chain():
    docs = []
    txt_loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    docs.extend(txt_loader.load())
    docs.extend(pdf_loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

qa_chain = initialize_qa_chain()

# API endpoint
@app.post("/answer", response_model=AnswerResponse)
async def answer_query(request: QueryRequest):
    try:
        result = qa_chain.invoke({"query": request.query})
        return {"answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
