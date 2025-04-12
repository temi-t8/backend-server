# Mohawk College Conversational Assistant 🏫

Welcome to the **Mohawk College Conversational Assistant**! This is a **FastAPI**-based backend service designed to handle user queries about Mohawk College programs, student services, and more. It uses **LangChain** for Retrieval-Augmented Generation (RAG), **OpenAI** for text processing, and **MongoDB** for storing conversation history.

## ✨ Key Features
- **Retrieval-Augmented Generation (RAG)** using local documents for more accurate, context-specific answers.  
- **OpenAI GPT** integration for query refinement and fallback responses.  
- **MongoDB Storage** for logging user queries and responses.  
- **WebSocket Support** for real-time interactive conversations.  
- **External Crawling** to fetch content from Mohawk College or MSA websites when queries pertain to specific keywords.  

## 📂 Table of Contents
1. [Prerequisites](#-prerequisites)
2. [Project Setup](#-project-setup)
3. [Environment Variables](#-environment-variables)
4. [Run the Application](#-run-the-application)
5. [API Endpoints](#-api-endpoints)
6. [WebSocket Usage](#-websocket-usage)
7. [Database Integration](#-database-integration)
8. [Directory Structure](#-directory-structure)
---

## 🛠 Prerequisites
- **Python 3.9+** (Recommended)
- **pip** or **conda** for installing Python packages
- **MongoDB** instance (local or remote)
- **An OpenAI API key** (for GPT-3.5-turbo or GPT-4 calls)
---

## 🏗 Project Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/temi-t8/backend-server.git
   ```
2. **Create a virtual environment (optional but recommended)**:
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. 🔐 Environment Variables
```bash
export OPENAI_API_KEY=sk-proj
export MONGODB_URI="mongodb+srv://TEMI:robot@"
```
NOTE: You can create your own MongoDB Cluster and add MongoDB URI but make sure to keep double inverted comma to avoid error

## 🚀 Run the Application
After installing all dependencies and setting your environment variables, you can start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
The application will be available at: http://localhost:8000

## 📡 API Endpoints
1. Health Check
- Endpoint: GET /health
- Description: Returns the current status of the service and MongoDB connection.
- Response Example:
  ```bash
  {
  "status": "running",
  "version": "1.0.0",
  "db_status": "healthy"
  }
  ```
2. Answer Query
- Endpoint: POST /answer
- Request Body:
  ```bash
  {
  "query": "What programs does Mohawk College offer related to Software?"
  }
  ```
  
- Response Example:
  ```bash
  {
  "answer": "Mohawk College offers the Computer Systems Technician – Software Development program..."
  }
  ```

- Description:
  - Refines the user query for spelling or partial program names.
  - Checks if the query relates to Mohawk programs, MSA, admissions, etc.
  - Crawls official Mohawk/related sites for matching content or uses Local RAG from the data folder to generate an answer.
  - If no context found, it uses a fallback with OpenAI GPT to provide a generic or playful response.

## ⚡ WebSocket Usage
- Endpoint: GET /ws
- Protocol: WebSocket

This endpoint allows real-time conversational interaction:

1. Connect to the WebSocket at ws://localhost:8000/ws.
2. Send a message (the user query).
3. Receive a text response.

Example (using websocat or similar CLI tool):
```bash
websocat ws://localhost:8000/ws
> Hello, what jokes do you know?
< Here's a fun one...
```
The server will:
- Refine your query.
- Check if it’s a fun/joke request or a Mohawk-related query.
- Respond with either RAG-based or fallback content.
- Store the conversation in MongoDB.

NOTE: if this server is being deployed anywhere it should use secure websocket such as `wss://domain/ws`/

## 🌐 Database Integration

This service uses MongoDB to record each user query and the corresponding answer.
- Collection Name: ConversationTracker
- Stored Fields:
  - query: The original user query.
  - answer: The assistant’s answer.
  - timestamp: DateTime of the stored record.

## 🗂 Directory Structure
```bash
.
├── data/
│   ├── faqs.txt
│   └── ...
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

If you have any questions, reach out via email me at `nevil-dineshkumar.patel@mohawkcollege.ca`
