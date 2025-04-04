import os
import logging
import difflib
import requests
import openai
from datetime import datetime
from typing import Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "RAGDB")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ConversationTracker")

# Set OpenAI key
openai.api_key = OPENAI_API_KEY

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App init
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# MongoDB Manager
class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None

    async def connect(self):
        self.client = AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        logger.info("Connected to MongoDB")

    async def store_conversation(self, query: str, answer: str):
        await self.collection.insert_one({"query": query, "answer": answer, "timestamp": datetime.utcnow()})

db_manager = DatabaseManager()

# Models
class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str

# Query Refinement
async def refine_query(original_query: str) -> str:
    system_prompt = (
        "You are a helpful assistant for Mohawk College. Fix spelling and recognition errors in the query. "
        "If the query refers to a Mohawk program, fix and complete it. Return only the corrected query."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_query}
            ]
        )
        return response["choices"][0]["message"]["content"].strip('"')
    except:
        return original_query

# Fallbacks
async def generate_fallback_response(query: str) -> str:
    fun_keywords = ["joke", "story", "riddle", "game"]
    if any(k in query.lower() for k in fun_keywords):
        prompt = "Tell a fun short joke, riddle, or story."
    else:
        prompt = "User asked something unrelated. Kindly redirect or entertain politely."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )
    return response["choices"][0]["message"]["content"].strip()

# Crawlers
async def get_program_page_content(query: str) -> Optional[str]:
    try:
        res = requests.get("https://www.mohawkcollege.ca/programs/search", timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        links = {
            a.text.strip(): a["href"]
            for a in soup.find_all("a", href=True)
            if a["href"].startswith("/programs/")
        }
        match = difflib.get_close_matches(query, list(links.keys()), n=1, cutoff=0.4)
        if match:
            url = "https://www.mohawkcollege.ca" + links[match[0]]
            res = requests.get(url, timeout=10)
            return BeautifulSoup(res.text, "html.parser").get_text(separator="\n", strip=True)[:5000]
    except Exception as e:
        logger.warning(f"[Program Crawler] {e}")
    return None

async def get_msa_or_services_content(query: str) -> Optional[str]:
    sources = [
        ("msa", "https://mohawkstudents.ca/"),
        ("student", "https://mohawkstudents.ca/"),
        ("admission", "https://www.mohawkcollege.ca/future-ready-toolkit/services-supports"),
        ("support", "https://www.mohawkcollege.ca/future-ready-toolkit/services-supports"),
    ]
    for keyword, url in sources:
        if keyword in query:
            try:
                res = requests.get(url, timeout=10)
                return BeautifulSoup(res.text, "html.parser").get_text(separator="\n", strip=True)[:5000]
            except Exception as e:
                logger.warning(f"[MSA/Services Crawler] {e}")
    return None

# RAG Engine
def initialize_rag():
    logger.info("Initializing RAG...")
    docs = DirectoryLoader("data", glob="**/*", loader_cls=TextLoader).load()
    chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff", retriever=vectordb.as_retriever()
    )

qa_chain = initialize_rag()

def run_rag_on_text(query: str, content: str) -> Optional[str]:
    doc = Document(page_content=content)
    chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents([doc])
    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff", retriever=vectordb.as_retriever()
    )
    result = rag_chain.invoke({"query": query})
    return result["result"]

# REST Endpoint
@app.post("/answer", response_model=AnswerResponse)
async def answer_query(request: QueryRequest):
    original = request.query.strip()
    refined = await refine_query(original)
    refined_lower = refined.lower()

    for keywords, fetch_func in [
        (["msa", "student", "club"], get_msa_or_services_content),
        (["program", "course", "mohawk", "college"], get_program_page_content),
        (["admission", "support", "registration"], get_msa_or_services_content),
    ]:
        if any(k in refined_lower for k in keywords):
            content = await fetch_func(refined_lower)
            if content:
                answer = run_rag_on_text(refined_lower, content)
                if answer:
                    await db_manager.store_conversation(original, answer)
                    return {"answer": answer}

    logger.info("Using local RAG...")
    result = qa_chain.invoke({"query": refined})
    answer = result["result"]
    if not answer or "i don't know" in answer.lower():
        fallback = await generate_fallback_response(refined)
        await db_manager.store_conversation(original, fallback)
        return {"answer": fallback}

    await db_manager.store_conversation(original, answer)
    return {"answer": answer}

# WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            refined = await refine_query(msg)
            refined_lower = refined.lower()

            for keywords, fetch_func in [
                (["msa", "student", "club"], get_msa_or_services_content),
                (["program", "course", "mohawk", "college"], get_program_page_content),
                (["admission", "support", "registration"], get_msa_or_services_content),
            ]:
                if any(k in refined_lower for k in keywords):
                    content = await fetch_func(refined_lower)
                    if content:
                        answer = run_rag_on_text(refined_lower, content)
                        if answer:
                            await db_manager.store_conversation(msg, answer)
                            await websocket.send_text(answer)
                            break
            else:
                result = qa_chain.invoke({"query": refined})
                answer = result["result"]
                if not answer.strip() or "i don't know" in answer.lower():
                    fallback = await generate_fallback_response(refined)
                    await db_manager.store_conversation(msg, fallback)
                    await websocket.send_text(fallback)
                else:
                    await db_manager.store_conversation(msg, answer)
                    await websocket.send_text(answer)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

# Health check
@app.get("/health")
async def health_check():
    db_status = "healthy" if db_manager.client else "unavailable"
    return {"status": "running", "version": "1.0.0", "db_status": db_status}

# Startup / Shutdown
@app.on_event("startup")
async def startup():
    await db_manager.connect()
    logger.info("Service started")

@app.on_event("shutdown")
async def shutdown():
    if db_manager.client:
        db_manager.client.close()
    logger.info("Service stopped")
