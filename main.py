from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import os
import requests
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
import difflib
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# Load environment variables (needed for local testing)
load_dotenv()

# MongoDB Connection
MONGO_URI = os.environ.get("MONGODB_URI")
DB_NAME = "RAGDB"
COLLECTION_NAME = "ConversationTracker"

# Validate MongoDB URI
if not MONGO_URI:
    raise RuntimeError("Missing MONGODB_URI environment variable")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


async def store_question_answer(query: str, answer: str):
    document = {
        "query": query,
        "answer": answer,
        "timestamp": datetime.utcnow()
    }
    await collection.insert_one(document)


# Validate API key from env
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response Models
class QueryRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str


# Program page crawler with debug
def get_program_page_content(query: str) -> str | None:
    base_url = "https://www.mohawkcollege.ca"
    search_url = f"{base_url}/programs/search"
    try:
        print(f"\nCrawling programs list: {search_url}", flush=True)
        res = requests.get(search_url, timeout=10)
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
            print(f"[MATCH] Found: {title} => {full_url}", flush=True)
            page_res = requests.get(full_url, timeout=10)
            detail_soup = BeautifulSoup(page_res.text, "html.parser")
            return detail_soup.get_text(separator="\n", strip=True)[:5000]
        print("No program match found", flush=True)
        return None
    except Exception as e:
        print(f"Error during program crawl: {e}", flush=True)
        return None


# MSA or Services crawler
def get_msa_or_services_content(query: str) -> str | None:
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
                print(f"Crawling {keyword} content: {url}", flush=True)
                res = requests.get(url, timeout=10)
                soup = BeautifulSoup(res.text, "html.parser")
                return soup.get_text(separator="\n", strip=True)[:5000]
            except Exception as e:
                print(f"Error crawling {url}: {e}", flush=True)
    return None


# Load static RAG
print("Loading local documents...", flush=True)


def initialize_qa_chain():
    docs = []
    txt_loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs.extend(txt_loader.load())
    docs.extend(pdf_loader.load())
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(chunks, embeddings)
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )


qa_chain = initialize_qa_chain()
print("Local documents loaded.", flush=True)

# # REST API
# @app.post("/answer", response_model=AnswerResponse)
# async def answer_query(request: QueryRequest):
#     query = request.query.lower()
#     print(f"\n🚀 [REST] Query: {query}", flush=True)

#     if any(word in query for word in ["msa", "club", "event", "student association"]):
#         msa_text = get_msa_or_services_content(query)
#         if msa_text:
#             print("[REST] Using MSA/services crawler", flush=True)
#             return run_rag_on_text(query, msa_text)

#     elif "program" in query or "mohawk college" in query:
#         program_text = get_program_page_content(query)
#         if program_text:
#             print("[REST] Using program crawler", flush=True)
#             return run_rag_on_text(query, program_text)

#     elif any(word in query for word in ["admission", "registration", "support", "services"]):
#         services_text = get_msa_or_services_content(query)
#         if services_text:
#             print("📄 [REST] Using services crawler", flush=True)
#             return run_rag_on_text(query, services_text)

#     print("[REST] Using local documents", flush=True)
#     result = qa_chain.invoke({"query": request.query})
#     return {"answer": result["result"]}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()
            lowered = query.lower()
            print(f"\n[WebSocket] Query: {lowered}", flush=True)

            if any(word in lowered for word in ["msa", "club", "event", "student association"]):
                msa_text = get_msa_or_services_content(lowered)
                if msa_text:
                    print("[WebSocket] Using MSA/services crawler", flush=True)
                    result = run_rag_on_text(lowered, msa_text)
                    await store_question_answer(lowered, result["answer"])
                    await websocket.send_text(result["answer"])
                    continue

            elif "program" in lowered or "mohawk college" in lowered:
                program_text = get_program_page_content(lowered)
                if program_text:
                    print("[WebSocket] Using program crawler", flush=True)
                    result = run_rag_on_text(lowered, program_text)
                    await store_question_answer(lowered, result["answer"])
                    await websocket.send_text(result["answer"])
                    continue

            elif any(word in lowered for word in ["admission", "registration", "support", "services"]):
                services_text = get_msa_or_services_content(lowered)
                if services_text:
                    print("[WebSocket] Using services crawler", flush=True)
                    result = run_rag_on_text(lowered, services_text)
                    await store_question_answer(lowered, result["answer"])
                    await websocket.send_text(result["answer"])
                    continue

            print("[WebSocket] Using local RAG", flush=True)
            result = qa_chain.invoke({"query": query})
            await store_question_answer(query, result["result"])
            await websocket.send_text(result["result"])

    except WebSocketDisconnect:
        print("WebSocket disconnected", flush=True)


# Reusable RAG on text
def run_rag_on_text(query: str, content: str) -> dict:
    doc = Document(page_content=content)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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
