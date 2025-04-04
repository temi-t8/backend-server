from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
import os
import openai
import requests
import difflib
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient

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

# --- Utilities ---
def is_domain_related(query_lower: str) -> bool:
    keywords = ["mohawk", "college", "campus", "student", "program", "course", "msa", "admission"]
    return any(word in query_lower for word in keywords)

async def refine_query(original_query: str) -> str:
    system_prompt = (
    "You are a helpful assistant for Mohawk College. Fix spelling and recognition errors in the query. "
    "If the query refers to a Mohawk program and the name is partially written or incorrect, correct and complete it. "
    "Example: 'sofware devlopment' should become 'Computer system Technician Software Development program at Mohawk College'. "
    "Return only the corrected query as a question."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_query}
            ]
        )
        refined = response["choices"][0]["message"]["content"].strip()
        print(f"[Refinement] '{original_query}' -> '{refined}'", flush=True)
        return refined.strip('"')
    except Exception as e:
        print(f"[Refinement Error] {e}", flush=True)
        return original_query

async def generate_fallback_response(refined_query: str, original_query: str) -> str:
    rq_lower = refined_query.lower()
    fun_triggers = [
        "joke", "tell me a joke", "funny", "story", "short story",
        "let's play", "play a game", "riddle", "entertain me", "make me laugh", "game"
    ]

    try:
        # Fun fallback
        if any(term in rq_lower for term in fun_triggers):
            print(f"[Fallback] 🎭 Fun fallback for query: {refined_query}", flush=True)
            prompt = (
                "You are a cheerful and witty assistant. The user wants to be entertained — "
                "respond with a short joke, funny story, riddle, or quick game in a fun and engaging way!"
            )
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": refined_query}
                ],
                temperature=0.9
            )
            return response["choices"][0]["message"]["content"].strip()

        # Inappropriate content handling (GPT detects and responds politely)
        print(f"[Fallback] 🔍 Checking for inappropriate or unsafe content...", flush=True)
        moderation_prompt = (
            "You are a safety-conscious assistant. If the user's message includes unsafe, offensive, or inappropriate content, "
            "respond in a calm, respectful, and polite way. Do not directly accuse or label the user. "
            "Keep your response short and friendly, and encourage respectful interaction."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": moderation_prompt},
                {"role": "user", "content": refined_query}
            ],
            temperature=0.7
        )
        answer = response["choices"][0]["message"]["content"].strip()

        # Heuristic: if the GPT thinks it's inappropriate, it will respond carefully
        if any(keyword in answer.lower() for keyword in ["inappropriate", "offensive", "not allowed", "respectful", "safe environment"]):
            print(f"[Fallback]Inappropriate response triggered: {refined_query}", flush=True)
            return answer

        # General fallback
        print(f"[Fallback] General fallback triggered for query: {refined_query}", flush=True)
        general_prompt = (
            "You are a helpful and engaging assistant for Mohawk College. The user's question is outside your scope, "
            "so respond politely and redirect them to ask something related to the college, or share something fun if appropriate."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": general_prompt},
                {"role": "user", "content": refined_query}
            ],
            temperature=0.5
        )
        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"[Fallback Error] {e}", flush=True)
        return "Oops! Something went wrong while answering that. Try again later."

# --- Crawlers ---
async def get_program_page_content(query: str) -> Optional[str]:
    base_url = "https://www.mohawkcollege.ca"
    try:
        print("[Crawler] Searching Mohawk program page...", flush=True)
        res = requests.get(f"{base_url}/programs/search", timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        links = {
            a.text.strip(): a["href"]
            for a in soup.find_all("a", href=True) if a["href"].startswith("/programs/")
        }
        match = difflib.get_close_matches(query, list(links.keys()), n=1, cutoff=0.4)
        if match:
            detail_url = base_url + links[match[0]]
            res = requests.get(detail_url, timeout=10)
            return BeautifulSoup(res.text, "html.parser").get_text(separator="\n", strip=True)[:5000]
    except Exception as e:
        print(f"[Crawler Error - Program] {e}", flush=True)
    return None

async def get_msa_or_services_content(query: str) -> Optional[str]:
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
                print(f"[Crawler] Crawling for '{keyword}' at {url}", flush=True)
                res = requests.get(url, timeout=10)
                return BeautifulSoup(res.text, "html.parser").get_text(separator="\n", strip=True)[:5000]
            except Exception as e:
                print(f"[Crawler Error - MSA/Services] {e}", flush=True)
    return None

# --- Local RAG ---
def initialize_qa_chain():
    print("[Init] Loading local documents...", flush=True)
    docs = []
    docs.extend(DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader).load())
    docs.extend(DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader).load())
    chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff", retriever=vectordb.as_retriever()
    )
qa_chain = initialize_qa_chain()

def run_rag_on_text(query: str, content: str) -> dict:
    doc = Document(page_content=content)
    chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents([doc])
    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents(query)
    if not docs:
        print("[RAG] No relevant documents found", flush=True)
        return {"fallback": True, "answer": None}
    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff", retriever=retriever
    )
    result = rag_chain.invoke({"query": query})
    return {"fallback": False, "answer": result["result"]}

# --- API Endpoints ---
@app.post("/answer", response_model=AnswerResponse)
async def answer_query(request: QueryRequest):
    original = request.query.strip()
    refined = await refine_query(original)
    refined_lower = refined.lower()

    for checker, fetch_func in [
        (["msa", "club", "student", "association"], get_msa_or_services_content),
        (["program", "course", "mohawk", "college"], get_program_page_content),
        (["admission", "registration", "support", "services"], get_msa_or_services_content)
    ]:
        if any(w in refined_lower for w in checker):
            content = await fetch_func(refined_lower)
            if content:
                rag_result = run_rag_on_text(refined_lower, content)
                if not rag_result["fallback"]:
                    return {"answer": rag_result["answer"]}

    print("[REST] Using local RAG fallback...", flush=True)
    result = qa_chain.invoke({"query": refined})
    if not result["result"].strip():
        fallback = await generate_fallback_response(refined, original)
        return {"answer": fallback}
    return {"answer": result["result"]}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            print(f"\n[WebSocket] Original Query: {msg}", flush=True)
            refined = await refine_query(msg)
            refined_lower = refined.lower()

            for checker, fetch_func in [
                (["msa", "club", "student", "association"], get_msa_or_services_content),
                (["program", "course", "mohawk", "college"], get_program_page_content),
                (["admission", "registration", "support", "services"], get_msa_or_services_content)
            ]:
                if any(w in refined_lower for w in checker):
                    content = await fetch_func(refined_lower)
                    if content:
                        rag_result = run_rag_on_text(refined_lower, content)
                        if not rag_result["fallback"]:
                            await db_manager.store_conversation(msg, rag_result["answer"])
                            await websocket.send_json({"answer": rag_result["answer"]})
                            break
            else:
                print("[WebSocket] Using local RAG...", flush=True)
                result = qa_chain.invoke({"query": refined})
                if not result["result"].strip() or "i don't know" in result["result"].lower():
                    print("🔁 Triggering fallback: No good local RAG response", flush=True)
                    fallback = await generate_fallback_response(refined, msg)
                    await db_manager.store_conversation(msg, fallback)
                    await websocket.send_json({"answer": fallback})
                else:
                    await websocket.send_json({"answer": result["result"]})
                    await db_manager.store_conversation(msg, result["result"])
    except WebSocketDisconnect:
        print("WebSocket disconnected", flush=True)

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