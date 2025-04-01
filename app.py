import os
import logging
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
import difflib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
MONGO_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = os.getenv("DB_NAME", "RAGDB")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ConversationTracker")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

if not MONGO_URI or not OPENAI_API_KEY:
    logger.critical("Missing required environment variables")
    raise RuntimeError("Missing required environment variables")

# Constants
REQUEST_TIMEOUT = 10
MAX_RESPONSE_LENGTH = 5000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Initialize FastAPI
app = FastAPI(title="RAG WebSocket Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Database Manager
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

# Web Scraper
class WebScraper:
    @staticmethod
    async def scrape_page(url: str) -> Optional[str]:
        try:
            res = requests.get(url, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            return BeautifulSoup(res.text, "html.parser").get_text(separator="\n", strip=True)[:MAX_RESPONSE_LENGTH]
        except requests.RequestException as e:
            logger.error(f"Scraping error: {e}")
            return None

    @staticmethod
    async def scrape_program_page(query: str) -> Optional[str]:
        base_url = "https://www.mohawkcollege.ca"
        search_url = f"{base_url}/programs/search"
        res = requests.get(search_url, timeout=REQUEST_TIMEOUT)
        soup = BeautifulSoup(res.text, "html.parser")
        program_links = {link.text.strip(): link["href"] for link in soup.find_all("a", href=True) if link["href"].startswith("/programs/")}
        match = difflib.get_close_matches(query, list(program_links.keys()), n=1, cutoff=0.4)
        if match:
            return await WebScraper.scrape_page(base_url + program_links[match[0]])
        return None

# RAG Service
class RAGService:
    def __init__(self):
        self.qa_chain = None

    async def initialize(self):
        docs = DirectoryLoader("data", glob="**/*", loader_cls=TextLoader).load()
        chunks = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(docs)
        vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
        self.qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY), retriever=vectordb.as_retriever())
        logger.info("RAG initialized")

    async def query(self, question: str) -> str:
        return self.qa_chain.invoke({"query": question})["result"]

rag_service = RAGService()

# WebSocket API
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            query = await websocket.receive_text()
            lowered = query.lower()
            response = None

            if any(word in lowered for word in ["msa", "club", "event", "student association"]):
                response = await WebScraper.scrape_page("https://mohawkstudents.ca/")
            elif any(word in lowered for word in ["admission", "registration", "support", "services"]):
                response = await WebScraper.scrape_page("https://www.mohawkcollege.ca/future-ready-toolkit/services-supports")
            elif "program" in lowered or "mohawk college" in lowered:
                response = await WebScraper.scrape_program_page(lowered)

            if response:
                answer = await rag_service.query(response)
            else:
                answer = await rag_service.query(query)

            await db_manager.store_conversation(query, answer)
            await websocket.send_text(answer)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
            break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.send_text("Error processing request")

# API Endpoint for Health Check
@app.get("/health")
async def health_check():
    return {"status": "running", "version": app.version, "db_status": "healthy"}

# Startup Event
@app.on_event("startup")
async def startup():
    await db_manager.connect()
    await rag_service.initialize()
    logger.info("Service started")

# Shutdown Event
@app.on_event("shutdown")
async def shutdown():
    if db_manager.client:
        db_manager.client.close()
    logger.info("Service stopped")
