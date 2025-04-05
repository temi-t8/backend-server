from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
import os
from openai import OpenAI

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
import re
from collections import Counter
from difflib import SequenceMatcher
from langchain.prompts import PromptTemplate
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
client = OpenAI(api_key=OPENAI_API_KEY)

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
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a friendly and knowledgeable assistant for Mohawk College. 
Answer the question using the provided information below. Be engaging, helpful, and concise. 
Use a cheerful tone when appropriate.

Context:
{context}

Question:
{question}
"""
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
        "You are a helpful assistant for Mohawk College. Your job is to fix minor spelling or recognition errors in the query. "
        "If it's a joke, story, fun question, or out-of-scope query, or weather related or time, leave it mostly untouched. "
        "Also, do NOT respond to the query yourself. Only return the refined query as-is, just corrected."
        "If the query refers to a Mohawk program and the name is partially written or incorrect, correct and complete it. "
        "Example: 'sofware devlopment' should become 'Computer system Technician Software Development program at Mohawk College'. "
    )

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_query}
        ])
        refined = response.choices[0].message.content.strip()
        print(f"[Refinement] '{original_query}' -> '{refined}'", flush=True)
        return refined.strip('"')
    except Exception as e:
        print(f"[Refinement Error] {e}", flush=True)
        return original_query

async def generate_fallback_response(refined_query: str, original_query: str) -> str:
    rq_lower = refined_query.lower()
    fun_triggers = [
        "joke", "tell me a joke", "funny", "story", "short story",
        "let's play", "play a game", "riddle", "entertain me", "make me laugh", "game", "laugh", "happy"
    ]

    time_weather_triggers = [
        "weather", "temperature", "forecast", "raining", "sunny",
        "what time", "time is it", "current time", "clock", "date", "day"
    ]

    try:
        # --- Fun Queries ---
        if any(term in rq_lower for term in fun_triggers):
            print(f"[Fallback] 🎭 Fun fallback for: {refined_query}", flush=True)

            try:
                prompt = (
                    "You are a cheerful assistant. The user wants something fun. "
                    "Tell a short joke, funny story, or riddle in a playful tone."
                )
                response = client.chat.completions.create(model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": refined_query}
                ],
                temperature=0.9)
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Fun Fallback Error] {e}", flush=True)
                return "I'm here to entertain you, but I ran into an issue. Try again in a bit!"

        # --- Time/Weather Queries ---
        if any(term in rq_lower for term in time_weather_triggers):
            print(f"[Fallback] Time/Weather fallback triggered: {refined_query}", flush=True)

            try:
                prompt = (
                    "You are a playful weather assistant for college students in Ontario. If you can't access real weather, pretend to be a fun weather forecaster and give a cheerful imaginary forecast."
                )
                response = client.chat.completions.create(model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": refined_query}
                ],
                temperature=0.8)
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Time/Weather Fallback Error] {e}", flush=True)
                return "I'd love to help, but I don't have a window or a clock... just vibes!"

        # --- Inappropriate Detection (Polite Response) ---
        print(f"[Fallback] 🔍 Checking content safety...", flush=True)
        try:
            moderation_prompt = (
                "You are a safe and polite assistant. If the query seems offensive or inappropriate, "
                "reply kindly and maintain respectful tone. Otherwise, say it's okay."
            )
            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": moderation_prompt},
                {"role": "user", "content": refined_query}
            ],
            temperature=0.3)
            answer = response.choices[0].message.content.strip()
            if any(word in answer.lower() for word in ["inappropriate", "offensive", "not allowed", "respectful", "violates"]):
                print(f"[Fallback] ❌ Inappropriate content blocked", flush=True)
                return answer
        except Exception as e:
            print(f"[Moderation Error] {e}", flush=True)

        # --- General Fallback ---
        print(f"[Fallback] ⚠️ General fallback triggered", flush=True)
        try:
            prompt = (
                "You are a helpful assistant for Mohawk College. The user asked something out of scope. "
                "Gently redirect them or share something fun, helpful, or generic if possible."
            )
            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": refined_query}
            ],
            temperature=0.5)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[General Fallback Error] {e}", flush=True)
            return "I'm not sure how to answer that, but I'm happy to help with anything Mohawk-related 😊"

    except Exception as e:
        print(f"[Fallback Crash] {e}", flush=True)
        return "Oops! Something went wrong while trying to respond. Try again shortly."



async def get_program_page_content(query: str) -> Optional[str]:
    base_url = "https://www.mohawkcollege.ca"
    search_url = f"{base_url}/programs/search"
    try:
        print("[Crawler] Fetching all program titles...", flush=True)
        res = requests.get(search_url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Build dictionary of {title: href}
        programs = {
            a.text.strip(): a["href"]
            for a in soup.find_all("a", href=True)
            if a["href"].startswith("/programs/") and a.text.strip()
        }

        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))

        # Score each program based on keyword matches
        matched_titles = []
        for title in programs.keys():
            title_words = set(re.findall(r'\b\w+\b', title.lower()))
            if query_keywords & title_words:
                matched_titles.append(title)

        if matched_titles:
            print(f"[Crawler] Found {len(matched_titles)} matched programs", flush=True)
            # Summarize list via ChatGPT
            system_prompt = (
                "You are a friendly assistant for Mohawk College. The user asked for a list of programs. "
                "Present a helpful and engaging response with matching program names."
            )
            user_prompt = (
                f"The user asked: '{query}'"
            )

            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7)
            return response.choices[0].message.content.strip()

        print("[Crawler] No matching program titles found", flush=True)

    except Exception as e:
        print(f"[Crawler Error - Program List] {e}", flush=True)

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
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": rag_prompt}
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
        (["programs", "program","course", "mohawk", "college"], get_program_page_content),
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

            # --- FUN FALLBACK FIRST ---
            if any(term in refined_lower for term in [
                "joke", "tell me a joke", "riddle", "story", "funny", "laugh", "game", "entertain"
            ]):
                print("[WebSocket] 🎭 Fun fallback triggered before RAG", flush=True)
                fallback = await generate_fallback_response(refined, msg)
                clean_fallback = fallback.replace("\n", " ").strip()
                await db_manager.store_conversation(msg, clean_fallback)
                await websocket.send_json({"answer": clean_fallback})
                continue

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
                            clean_answer = rag_result["answer"].replace("\n", " ").strip()
                            await db_manager.store_conversation(msg, clean_answer)
                            await websocket.send_json({"answer": clean_answer})
                            break
            else:
                print("[WebSocket] Using local RAG...", flush=True)
                result = qa_chain.invoke({"query": refined})
                if not result["result"].strip() or "i don't know" in result["result"].lower():
                    print("🔁 Triggering fallback: No good local RAG response", flush=True)
                    fallback = await generate_fallback_response(refined, msg)
                    clean_fallback = fallback.replace("\n", " ").strip()
                    await db_manager.store_conversation(msg, clean_fallback)
                    await websocket.send_json({"answer": clean_fallback})
                else:
                    clean_answer = result["result"].replace("\n", " ").strip()
                    await websocket.send_json({"answer": clean_answer})
                    await db_manager.store_conversation(msg, clean_answer)
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
