# 🎓 SECE RAG AI Chatbot

An intelligent AI-powered chatbot designed for **Sri Eshwar College of Engineering** that automatically scrapes college website data, transforms it into vector embeddings, and provides accurate answers using a **Retrieval-Augmented Generation (RAG)** pipeline powered by **Groq LLMs**.

---

## 📌 Project Overview

Students and visitors often struggle to find specific information hidden across multiple pages of a college website. This project solves that problem by creating an AI assistant that understands the entire college website and provides instant conversational responses.

The system performs automated web scraping, processes the collected data into embeddings, stores them in a vector database, and retrieves relevant information during user queries before generating a response using Groq's high-speed language models.

---

## ✨ Features

- 🌐 Automated college website scraping
- 🔎 Deep crawling of multiple web pages
- 📄 Intelligent text extraction and cleaning
- 🧩 Document chunking for better retrieval
- 🧠 Embedding generation using local embedding models
- 🗃️ Vector storage using ChromaDB
- ⚡ Fast response generation using Groq LLM
- 🤖 RAG-based context-aware question answering
- 🔄 Rebuild and update knowledge base
- 🎨 User-friendly web interface
- 🔍 Retrieval inspector to view retrieved chunks

---

## 🏗️ System Architecture

```
College Website
        |
        ↓
 Web Scraper / Crawler
        |
        ↓
 Data Cleaning & Processing
        |
        ↓
 Document Chunking
        |
        ↓
 Embedding Generation
        |
        ↓
 ChromaDB Vector Store
        |
        ↓
 User Query
        |
        ↓
 Similarity Search
        |
        ↓
 Relevant Context Retrieval
        |
        ↓
 Groq LLM
        |
        ↓
 AI Generated Response
```

---

## 🛠️ Tech Stack

### Backend
- Python
- FastAPI / Flask (depending on implementation)

### AI & RAG
- Groq LLM
- LangChain
- ChromaDB
- Sentence Transformers / Local Embeddings

### Web Scraping
- BeautifulSoup
- Requests
- Web Crawlers

### Frontend
- HTML
- CSS
- JavaScript

---

## 🚀 Workflow

### 1. Data Collection
The crawler visits the Sri Eshwar College website and extracts meaningful information from multiple pages.

### 2. Data Processing
The collected content is cleaned, split into smaller chunks, and prepared for embedding generation.

### 3. Vector Database Creation
The chunks are converted into numerical embeddings and stored inside ChromaDB for efficient semantic search.

### 4. User Query Processing
When a user asks a question, the system converts the query into embeddings and retrieves the most relevant chunks from ChromaDB.

### 5. AI Response Generation
The retrieved context is passed to Groq LLM, which generates a precise answer grounded in the college data.

---

## 📸 User Interface

The chatbot interface includes:

- Chat assistant panel
- Groq/Gemini model configuration
- API key management
- Index rebuilding options
- Deep website crawling
- Retrieval debugging panel

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/SECE-RAG-Chatbot.git
cd SECE-RAG-Chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### Run Application

```bash
python app.py
```

Open:

```
http://127.0.0.1:7861
```

---

## 📂 Project Structure

```
SECE-RAG-Chatbot/
│
├── scraper/            # Website crawling and data extraction
├── data/               # Scraped documents
├── embeddings/         # Generated vector embeddings
├── chroma_db/          # ChromaDB vector storage
├── backend/            # API and RAG logic
├── frontend/           # User interface files
├── app.py              # Main application entry point
├── requirements.txt    # Python dependencies
├── .env                # API keys
└── README.md
```

---

## 🔮 Future Enhancements

- Add voice-based interaction
- Support multiple colleges
- Add admission enquiry assistant
- Improve crawling with scheduled updates
- Add user authentication and chat history
- Deploy using Docker and cloud services

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository and submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Developed by **Aswin N S**  
B.E Artificial Intelligence & Data Science  
Sri Eshwar College of Engineering
