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
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# ========== Configuration ==========
# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Validate environment variables
try:
    MONGO_URI = os.environ["MONGODB_URI"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    DB_NAME = os.environ.get("DB_NAME", "RAGDB")
    COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "ConversationTracker")
    ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")
except KeyError as e:
    logger.critical(f"Missing required environment variable: {e}")
    raise

# Constants
REQUEST_TIMEOUT = 10
MAX_RESPONSE_LENGTH = 5000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# ========== Application Setup ==========
app = FastAPI(
    title="RAG WebSocket Server",
    version="1.0.0",
    docs_url=None,  # Disable docs in production
    redoc_url=None
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Database Setup ==========
class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None

    async def connect(self):
        try:
            self.client = AsyncIOMotorClient(
                MONGO_URI,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=100,
                minPoolSize=10
            )
            await self.client.admin.command('ping')
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            logger.info("MongoDB connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise HTTPException(
                status_code=503,
                detail="Service unavailable (database connection failed)"
            )

    async def store_conversation(self, query: str, answer: str) -> bool:
        try:
            document = {
                "query": query,
                "answer": answer,
                "timestamp": datetime.utcnow()
            }
            await self.collection.insert_one(document)
            return True
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            return False

db_manager = DatabaseManager()

# ========== Models ==========
class HealthResponse(BaseModel):
    status: str
    version: str
    db_status: str

# ========== Core Services ==========
class WebScraper:
    @staticmethod
    async def scrape_program_page(query: str) -> Optional[str]:
        base_url = "https://www.mohawkcollege.ca"
        search_url = f"{base_url}/programs/search"
        
        try:
            res = requests.get(search_url, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            
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
                
                page_res = requests.get(full_url, timeout=REQUEST_TIMEOUT)
                page_res.raise_for_status()
                
                detail_soup = BeautifulSoup(page_res.text, "html.parser")
                return detail_soup.get_text(separator="\n", strip=True)[:MAX_RESPONSE_LENGTH]
            
            return None
            
        except requests.RequestException as e:
            logger.error(f"Web request failed: {e}")
        except Exception as e:
            logger.error(f"Scraping error: {e}")
        return None

class RAGService:
    def __init__(self):
        self.qa_chain = None
        self.initialized = False

    async def initialize(self):
        try:
            # Load documents
            docs = []
            txt_loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
            pdf_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
            docs.extend(txt_loader.load())
            docs.extend(pdf_loader.load())

            # Process documents
            splitter = CharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)
            
            # Create vector store
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectordb = FAISS.from_documents(chunks, embeddings)
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    openai_api_key=OPENAI_API_KEY
                ),
                chain_type="stuff",
                retriever=vectordb.as_retriever(),
            )
            self.initialized = True
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
            raise

    async def query(self, question: str) -> dict:
        if not self.initialized:
            return {"answer": "Service unavailable", "success": False}
        
        try:
            result = self.qa_chain({"query": question})
            return {"answer": result["result"], "success": True}
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {"answer": "Error processing request", "success": False}

rag_service = RAGService()

# ========== API Endpoints ==========
@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        await db_manager.client.admin.command('ping')
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "running",
        "version": app.version,
        "db_status": db_status
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"
    logger.info(f"WebSocket connection from {client_ip}")
    
    try:
        while True:
            try:
                # Receive and validate query
                query = await websocket.receive_text()
                if not query or len(query) > 1000:
                    await websocket.send_text("Invalid input length (1-1000 chars allowed)")
                    continue
                
                # Process query
                lowered = query.lower()
                
                # Web scraping logic
                if any(word in lowered for word in ["msa", "club", "event"]):
                    content = await WebScraper.scrape_program_page(lowered)
                    if content:
                        result = await rag_service.query(content)
                        await db_manager.store_conversation(query, result["answer"])
                        await websocket.send_text(result["answer"])
                        continue
                
                # Standard RAG processing
                result = await rag_service.query(query)
                await db_manager.store_conversation(query, result["answer"])
                await websocket.send_text(result["answer"])
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                await websocket.send_text("An error occurred processing your request")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected ({client_ip})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

# ========== Startup/Shutdown ==========
@app.on_event("startup")
async def startup():
    try:
        await db_manager.connect()
        await rag_service.initialize()
        logger.info("Service startup completed")
    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    try:
        if db_manager.client:
            db_manager.client.close()
        logger.info("Service shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# ========== Main Execution ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_config=None  # Use default logging
    )
