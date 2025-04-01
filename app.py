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

# Web Scrapers
class WebScraper:

    @staticmethod
    def get_program_page_content(query: str) -> Optional[str]:
        base_url = "https://www.mohawkcollege.ca"
        search_url = f"{base_url}/programs/search"
        try:
            logger.info(f"Crawling programs list: {search_url}")
            res = requests.get(search_url, timeout=REQUEST_TIMEOUT)
            soup = BeautifulSoup(res.text, "html.parser")
            program_links = {
                link.text.strip(): link["href"]
                for link in soup.find_all("a", href=True)
                if link["href"].startswith("/programs/") and link.text.strip()
            }
            titles = list(program_links.keys())
            match = difflib.get_close_matches(query, titles, n=1, cutoff=0.4)
            if match:
                title = match[0]
                href = program_links[title]
                full_url = base_url + href
                logger.info(f"[MATCH] Found: {title} => {full_url}")
                page_res = requests.get(full_url, timeout=REQUEST_TIMEOUT)
                detail_soup = BeautifulSoup(page_res.text, "html.parser")
                return detail_soup.get_text(separator="\n", strip=True)[:MAX_RESPONSE_LENGTH]
            logger.info("No program match found")
            return None
        except Exception as e:
            logger.error(f"Error during program crawl: {e}")
            return None

    @staticmethod
    def get_msa_or_services_content(query: str) -> Optional[str]:
        sources = [
            ("msa", "https://mohawkstudents.ca/"),
            ("student association", "https://mohawkstudents.ca/"),
            ("club", "https://mohawkstudents.ca/"),
            ("admission", "https://www.mohawkcollege.ca/future-ready-toolkit/services-supports"),
            ("registration", "https://www.mohawkcollege.ca/future-ready-toolkit/services-supports"),
            ("support", "https://www.mohawkcollege.ca/future-ready-toolkit/services-supports"),
        ]
        for keyword, url in sources:
            if keyword in query:
                try:
                    logger.info(f"Crawling {keyword} content: {url}")
                    res = requests.get(url, timeout=REQUEST_TIMEOUT)
                    soup = BeautifulSoup(res.text, "html.parser")
                    return soup.get_text(separator="\n", strip=True)[:MAX_RESPONSE_LENGTH]
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
        return None

    @staticmethod
    def run_rag_on_text(query: str, content: str) -> dict:
        doc = Document(page_content=content)
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents([doc])
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()
        rag_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
            chain_type="stuff",
            retriever=retriever,
        )
        result = rag_chain.invoke({"query": query})
        return {"answer": result["result"]}


# WebSocket API
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()
            lowered = query.lower()
            logger.info(f"[WebSocket] Query: {lowered}")

            msa_text = WebScraper.get_msa_or_services_content(lowered)
            if msa_text:
                logger.info("[WebSocket] Using MSA/services crawler")
                result = WebScraper.run_rag_on_text(lowered, msa_text)
                await db_manager.store_conversation(lowered, result["answer"])
                await websocket.send_text(result["answer"])
                continue

            program_text = WebScraper.get_program_page_content(lowered)
            if program_text:
                logger.info("[WebSocket] Using program crawler")
                result = WebScraper.run_rag_on_text(lowered, program_text)
                await db_manager.store_conversation(lowered, result["answer"])
                await websocket.send_text(result["answer"])
                continue

            logger.info("[WebSocket] Using local RAG")
            answer = await rag_service.query(query)
            await db_manager.store_conversation(query, answer)
            await websocket.send_text(answer)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
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
