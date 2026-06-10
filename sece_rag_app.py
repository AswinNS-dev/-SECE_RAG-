import os
import math
import re
import time
import requests
import numpy as np
import faiss
import gradio as gr
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse
from google import genai as google_genai
from google.genai import types as genai_types
from groq import Groq
from sentence_transformers import SentenceTransformer

# Global Gemini client (initialized when API key is provided)
_gemini_client = None

# Constants
SECE_URL = "https://sece.ac.in"
TXT_FILE = "sece_knowledge_base.txt"
CRAWL_CACHE_FILE = "sece_crawled_pages.txt"
MAX_CRAWL_PAGES = math.inf # max p
SKIP_EXTENSIONS = {
    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
    '.zip', '.rar', '.tar', '.gz',
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.mp4', '.mp3', '.avi', '.mov', '.wmv',
    '.exe', '.msi', '.apk',
}

# Predefined detailed SECE Knowledge Base
SECE_PREDEFINED_KB = [
    """About Sri Eshwar College of Engineering (SECE):
Location: Coimbatore, Tamil Nadu, India.
TEA Code: The college code for Tamil Nadu Engineering Admissions (TNEA) is 2739 (TEA Code 2739).
Affiliation & Approval: Sri Eshwar College of Engineering (SECE) is affiliated to Anna University, Chennai, and is approved by the All India Council for Technical Education (AICTE), New Delhi. It is a premium engineering college known for academic excellence.""",

    """Departments at Sri Eshwar College of Engineering (SECE):
The college offers 12 specialized departments:
1. CSE-AIML: Computer Science and Engineering (Artificial Intelligence and Machine Learning)
2. AI&DS: Artificial Intelligence and Data Science
3. CSE: Computer Science and Engineering
4. Cyber Security: Computer Science and Engineering (Cyber Security)
5. CBS: Computer Science and Business Systems
6. CCE: Computer Science and Communication Engineering
7. ECE: Electronics and Communication Engineering
8. VLSI: Electronics and Communication Engineering (VLSI Design and Technology)
9. EEE: Electrical and Electronics Engineering
10. IT: Information Technology
11. MECH: Mechanical Engineering
12. S&H: Department of Science and Humanities (supporting first-year studies)""",

    """Admissions at Sri Eshwar College of Engineering (SECE):
Admissions Process: Admissions are open for the academic year 2025 (Admission 2025 open).
Counselling Route: Students can secure admissions through the Tamil Nadu Engineering Admissions (TNEA) single-window counselling using the TEA code 2739.
Management Quota: Seats are also available under the management quota for direct admissions for eligible candidates.""",

    """Academics at Sri Eshwar College of Engineering (SECE):
Curriculum: The college follows the regulations and curriculum prescribed by Anna University, Chennai. The active regulations implemented are R2021 and R2823 (R2021/R2823 curriculum).
ERP Portal: Academic records, attendance, internal test marks, and student profiles are managed via the Enterprise Resource Planning portal at erp.sece.ac.in.""",

    """Research and Development at Sri Eshwar College of Engineering (SECE):
Research Centre: SECE has a dedicated Centre for Research that coordinates academic and industrial research projects.
Intellectual Property Rights: The active IPR Cell assists faculty, researchers, and students in filing patents and commercializing innovations.
Grants & Projects: The institution regularly receives research grants, seed funds for novel proposals, and undertakes industry-sponsored consultancy projects.""",

    """Innovation and Startup Ecosystem at Sri Eshwar College of Engineering (SECE):
Innovation Hub: The college promotes creativity through the Centre for Innovation, Centres of Excellence (Cofs), and the Institution's Innovation Council (IIC).
Hackathons: Students participate in and host several national-level and state-level hackathons.
Ignite Startup Accelerator: The Ignite Startup Accelerator provides mentorship, infrastructure, and seed funding guidance to help students launch their startups directly from campus.""",

    """Training and Placements at Sri Eshwar College of Engineering (SECE):
Placements Cell: The Centre for Training & Placements handles career development.
Training Programs: SECE offers continuous training starting from the first year, including comprehensive aptitude training, programming skills development, mock interviews, and communication training.
Recruiters: Top recruiters from multi-national corporations (MNCs), product development firms, and core engineering industries recruit students from SECE each year.""",

    """International Relations and Alliances at Sri Eshwar College of Engineering (SECE):
Global Alliances: SECE has collaborated with reputed international universities and global research organizations.
Opportunities: Students can opt for semester abroad programs, international internships, and inward/outward student and faculty mobility programs to gain global experience.""",

    """Campus Facilities and Infrastructure at Sri Eshwar College of Engineering (SECE):
Infrastructure Details:
- Smart Classrooms: Digital and interactive classrooms with audio-visual equipment.
- Central Library: Fully automated library containing thousands of books, international journals, and digital library resources.
- Sports: Comprehensive outdoor playgrounds, running tracks, and indoor sports complexes.
- Hostel: Clean, safe separate hostels for boys and girls with hygienic food and high-speed Wi-Fi.
- Wi-Fi Campus: High-speed Wi-Fi connectivity across the entire campus.
- Medical: On-campus medical facility with a resident doctor and ambulance.""",

    """Contact and Portals for Sri Eshwar College of Engineering (SECE):
- Official Website: sece.ac.in
- Alumni Association Portal: alumni.sece.ac.in
- Virtual Campus Tour: campustour.sece.ac.in"""
]


def _normalize_url(url):
    """Normalize a URL: remove fragment, trailing slash, and query strings."""
    p = urlparse(url)
    # Keep scheme, netloc, path only (strip fragments and query params)
    clean = urlunparse((p.scheme, p.netloc, p.path.rstrip('/'), '', '', ''))
    return clean


def _is_sece_internal(url):
    """Returns True if the URL belongs to the sece.ac.in domain."""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc == 'sece.ac.in' or netloc.endswith('.sece.ac.in')
    except Exception:
        return False


def scrape_page(url, session, timeout=10):
    """Scrape a single page. Returns (page_text, list_of_internal_links)."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if res.status_code != 200:
            return None, []
        content_type = res.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            return None, []

        soup = BeautifulSoup(res.text, 'html.parser')

        # Collect internal links before stripping tags
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if not href or href.startswith('javascript') or href.startswith('mailto') or href.startswith('tel'):
                continue
            full_url = _normalize_url(urljoin(url, href))
            ext = os.path.splitext(urlparse(full_url).path)[1].lower()
            if ext in SKIP_EXTENSIONS:
                continue
            if _is_sece_internal(full_url):
                links.append(full_url)

        # Strip noisy tags
        for tag in soup(['script', 'style', 'noscript', 'meta', 'link']):
            tag.decompose()

        text = ' '.join(soup.get_text(separator=' ').split())
        return text if len(text) > 80 else None, links

    except Exception:
        return None, []


def crawl_sece_website(max_pages=MAX_CRAWL_PAGES, progress_fn=None):
    """
    BFS crawler that visits all internal pages of sece.ac.in.
    Returns combined text from all pages, labeled by URL.
    progress_fn: optional callable(current_count, max_pages, url) for progress updates.
    """
    session = requests.Session()
    visited = set()
    queue = deque([_normalize_url(SECE_URL)])
    all_texts = []
    page_count = 0

    print(f"Starting deep crawl of {SECE_URL} (max {max_pages} pages)...")

    while queue and page_count < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        # Skip non-HTML by extension
        ext = os.path.splitext(urlparse(url).path)[1].lower()
        if ext in SKIP_EXTENSIONS:
            continue

        text, links = scrape_page(url, session)

        if text:
            page_count += 1
            page_label = f"[Source Page: {url}]"
            all_texts.append(f"{page_label}\n{text}")
            if progress_fn:
                progress_fn(page_count, max_pages, url)
            else:
                print(f"  [{page_count}/{max_pages}] {url[:80]}")

        # Enqueue new links
        for link in links:
            if link not in visited:
                queue.append(link)

        # Small polite delay to avoid hammering the server
        time.sleep(0.1)

    print(f"Deep crawl complete. Scraped {page_count} pages from sece.ac.in.")
    return '\n\n'.join(all_texts)


def load_crawl_cache():
    """Load previously crawled data from disk cache."""
    if os.path.exists(CRAWL_CACHE_FILE):
        with open(CRAWL_CACHE_FILE, 'r', encoding='utf-8') as f:
            data = f.read()
        if data.strip():
            print(f"Loaded crawl cache from {CRAWL_CACHE_FILE} ({len(data):,} chars).")
            return data
    return None


def save_crawl_cache(text):
    """Save crawled text to disk for future runs."""
    with open(CRAWL_CACHE_FILE, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Crawl cache saved to {CRAWL_CACHE_FILE} ({len(text):,} chars).")



def chunk_text(text, chunk_size=150, overlap=20):
    """Chunks the text into 'chunk_size' words with 'overlap' words overlap."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    
    if len(words) <= chunk_size:
        return [" ".join(words)]
        
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
            
    return chunks


def prepare_knowledge_base(use_deep_crawl=True, force_recrawl=False, progress_fn=None):
    """
    Prepares and returns knowledge base text chunks.
    use_deep_crawl: if True, crawl all pages (or use cache).
    force_recrawl: if True, ignore cache and re-crawl the full site.
    """
    all_chunks = []

    # 1. Always include predefined KB
    for item in SECE_PREDEFINED_KB:
        all_chunks.extend(chunk_text(item, chunk_size=150, overlap=20))

    # 2. Web content
    web_text = None
    if use_deep_crawl:
        if not force_recrawl:
            web_text = load_crawl_cache()
        if web_text is None:
            web_text = crawl_sece_website(max_pages=MAX_CRAWL_PAGES, progress_fn=progress_fn)
            if web_text:
                save_crawl_cache(web_text)
    else:
        # Fallback: homepage only
        try:
            session = requests.Session()
            text, _ = scrape_page(SECE_URL, session)
            web_text = text
        except Exception:
            pass

    if web_text:
        all_chunks.extend(chunk_text(web_text, chunk_size=150, overlap=20))

    print(f"Total chunks created: {len(all_chunks)}")

    # Save chunks to file
    with open(TXT_FILE, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(all_chunks):
            f.write(f"--- Chunk {i} ---\n{chunk}\n\n")

    return all_chunks



class HybridEmbeddings:
    """Supports local HuggingFace embeddings or Gemini cloud embeddings (new google.genai SDK)."""
    GEMINI_EMBEDDING_MODELS = [
        "text-embedding-004",
        "gemini-embedding-exp-03-07",
        "embedding-001",
    ]

    def __init__(self, mode="local"):
        self.mode = mode
        self.gemini_model = "text-embedding-004"
        self.local_model = None

    def set_mode(self, mode):
        self.mode = mode
        if mode == "local" and self.local_model is None:
            print("Loading local SentenceTransformer (all-MiniLM-L6-v2)...")
            self.local_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed_with_gemini(self, texts, task_type):
        """Try each Gemini embedding model in order until one works."""
        global _gemini_client
        if _gemini_client is None:
            raise RuntimeError("Gemini client not initialised. Verify API key first.")

        task_enum = (
            genai_types.TaskType.RETRIEVAL_QUERY if task_type == "retrieval_query"
            else genai_types.TaskType.RETRIEVAL_DOCUMENT
        )

        content_list = [texts] if isinstance(texts, str) else texts
        last_error = None

        for model_name in self.GEMINI_EMBEDDING_MODELS:
            try:
                result = _gemini_client.models.embed_content(
                    model=model_name,
                    contents=content_list,
                    config=genai_types.EmbedContentConfig(task_type=task_enum)
                )
                self.gemini_model = model_name  # remember what worked
                vecs = [emb.values for emb in result.embeddings]
                return np.array(vecs, dtype="float32")
            except Exception as e:
                print(f"Embedding model '{model_name}' failed: {e}")
                last_error = e

        raise RuntimeError(f"All Gemini embedding models failed. Last error: {last_error}")

    def get_embeddings(self, texts, is_query=False):
        if self.mode == "local":
            if self.local_model is None:
                self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = self.local_model.encode(
                [texts] if isinstance(texts, str) else texts,
                convert_to_numpy=True
            )
            arr = np.array(embeddings, dtype="float32")
            # Return 1-D for single string
            return arr[0] if isinstance(texts, str) else arr
        else:
            task_type = "retrieval_query" if is_query else "retrieval_document"
            arr = self._embed_with_gemini(texts, task_type)
            return arr[0] if isinstance(texts, str) else arr


class HybridVectorStore:
    """FAISS index wrapper that caches separate indices for local vs. cloud embeddings."""
    def __init__(self, embedding_mode="local"):
        self.embeddings = HybridEmbeddings(mode=embedding_mode)
        self.chunks = []
        self.index = None

    def get_index_paths(self):
        suffix = "local" if self.embeddings.mode == "local" else "gemini"
        return f"sece_faiss_{suffix}.index", f"sece_chunks_{suffix}.npy"

    def build_index(self, chunks):
        self.chunks = chunks
        if not chunks:
            print("No chunks provided.")
            return
            
        print(f"Computing embeddings ({self.embeddings.mode} mode)...")
        embs = self.embeddings.get_embeddings(chunks, is_query=False)
        
        # Normalize for Cosine Similarity
        faiss.normalize_L2(embs)
        
        dimension = embs.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embs)
        
        index_path, chunks_path = self.get_index_paths()
        faiss.write_index(self.index, index_path)
        np.save(chunks_path, np.array(self.chunks, dtype=object))
        print(f"FAISS index saved to {index_path}.")

    def load_index(self):
        index_path, chunks_path = self.get_index_paths()
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            self.chunks = np.load(chunks_path, allow_pickle=True).tolist()
            print(f"Loaded FAISS index from {index_path}.")
            return True
        return False

    def search(self, query, k=4):
        if self.index is None:
            return []
            
        query_emb = self.embeddings.get_embeddings(query, is_query=True)
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)
            
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results


class SECEChatbotEngine:
    """RAG engine that supports Groq LLM + HF embeddings, or Gemini Cloud."""
    def __init__(self):
        self.mode = "groq"  # "groq" or "gemini"
        self.vector_store = HybridVectorStore(embedding_mode="local")
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        self.gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")
        self.api_configured = False

    def configure_engine(self, mode, api_key):
        global _gemini_client
        if not api_key:
            return False, "API key cannot be empty."

        api_key = api_key.strip()
        if mode == "groq":
            try:
                test_client = Groq(api_key=api_key)
                test_client.models.list()
                self.groq_api_key = api_key
                os.environ["GROQ_API_KEY"] = api_key
                self.mode = "groq"
                self.vector_store.embeddings.set_mode("local")
                self.api_configured = True

                if not self.vector_store.load_index():
                    print("Index not found. Generating local FAISS database...")
                    chunks = prepare_knowledge_base()
                    self.vector_store.build_index(chunks)
                return True, "Groq RAG Engine configured successfully!"
            except Exception as e:
                return False, f"Failed to verify Groq Key: {e}"
        else:
            try:
                # Initialise new google.genai client
                _gemini_client = google_genai.Client(api_key=api_key)

                # Quick probe — try each embedding model until one succeeds
                probe_models = [
                    "text-embedding-004",
                    "gemini-embedding-exp-03-07",
                    "embedding-001",
                ]
                probe_ok = False
                for m in probe_models:
                    try:
                        _gemini_client.models.embed_content(
                            model=m,
                            contents=["test"],
                            config=genai_types.EmbedContentConfig(
                                task_type=genai_types.TaskType.RETRIEVAL_QUERY
                            )
                        )
                        self.vector_store.embeddings.gemini_model = m
                        probe_ok = True
                        print(f"Gemini embedding probe succeeded with model: {m}")
                        break
                    except Exception as probe_err:
                        print(f"Probe failed for '{m}': {probe_err}")

                if not probe_ok:
                    return False, "No supported Gemini embedding model found for this API key."

                self.gemini_api_key = api_key
                os.environ["GOOGLE_API_KEY"] = api_key
                self.mode = "gemini"
                self.vector_store.embeddings.set_mode("gemini")
                self.api_configured = True

                if not self.vector_store.load_index():
                    print("Index not found. Generating Gemini FAISS database...")
                    chunks = prepare_knowledge_base()
                    self.vector_store.build_index(chunks)
                return True, f"Gemini Cloud RAG Engine configured (model: {self.vector_store.embeddings.gemini_model})!"
            except Exception as e:
                return False, f"Failed to verify Gemini Key: {e}"

    def auto_init_from_env(self):
        """Auto loads active keys from environment if available."""
        groq_key = os.environ.get("GROQ_API_KEY", self.groq_api_key)
        gemini_key = os.environ.get("GOOGLE_API_KEY", self.gemini_api_key)
        if groq_key:
            success, msg = self.configure_engine("groq", groq_key)
            if success:
                print(f"Auto-configured: Groq mode.")
                return
            print(f"Auto-configure Groq failed: {msg}")
        if gemini_key:
            success, msg = self.configure_engine("gemini", gemini_key)
            print(f"Auto-configure Gemini: {msg}")

    def generate_answer(self, query):
        if not self.api_configured:
            return "Please configure your API Key in the settings first.", []
            
        retrieved = self.vector_store.search(query, k=4)
        if not retrieved:
            return "No knowledge base index found. Please rebuild index.", []
            
        context_str = "\n\n".join([f"[Source Chunk {i+1} - Similarity: {score:.4f}]\n{text}" for i, (text, score) in enumerate(retrieved)])
        
        prompt = f"""You are an assistant for Sri Eshwar College of Engineering (SECE), Coimbatore.
Answer the user's question ONLY using the retrieved context provided below.
If the answer cannot be found or answered using ONLY the retrieved context, you MUST output exactly:
"I don't have information about this in my data. Please refer to sece.ac.in for official details."

Do not use outside knowledge or make up any details.

Retrieved Context:
{context_str}

Question: {query}

Answer:"""

        if self.mode == "groq":
            try:
                client = Groq(api_key=self.groq_api_key)
                completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.0
                )
                answer = completion.choices[0].message.content.strip()
                return answer, retrieved
            except Exception as e:
                return f"Error calling Groq LLM: {e}", retrieved
        else:
            try:
                global _gemini_client
                response = _gemini_client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(temperature=0.0)
                )
                answer = response.text.strip()
                return answer, retrieved
            except Exception as e:
                return f"Error calling Gemini LLM: {e}", retrieved


engine = SECEChatbotEngine()

SAMPLE_QUESTIONS = [
    "What is the TNEA code for Sri Eshwar College of Engineering and which university is it affiliated with?",
    "Which departments and engineering courses are offered at SECE?",
    "Can you tell me about the admissions process and if there is management quota?",
    "What research facilities, grants, and startup support are available at SECE?",
    "What are the top recruiters and placement support offered by the Centre for Training & Placements?"
]

custom_css = """
body { background-color: #0b0f19; font-family: 'Outfit', 'Inter', sans-serif; color: #f1f5f9; }
.sece-header { text-align: center; padding: 24px; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; border-bottom: 4px solid #f59e0b; margin-bottom: 24px; }
.sece-header h1 { color: #f59e0b !important; font-weight: 800; margin: 0; font-size: 2.2rem; }
.sece-header p { color: #94a3b8 !important; font-size: 1.1rem; margin-top: 8px; }
.chatbot-col { background-color: #111827; border-radius: 12px; padding: 16px; border: 1px solid #1f2937; }
.sidebar-col { background-color: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; }
.gr-button-primary { background: linear-gradient(to right, #d97706, #f59e0b) !important; border: none !important; color: white !important; font-weight: 600 !important; }
.sources-card { background-color: #0f172a; border-left: 4px solid #f59e0b; padding: 12px; margin-bottom: 12px; border-radius: 0 8px 8px 0; }
.sources-card h4 { margin: 0 0 6px 0; color: #fbbf24; font-size: 0.95rem; }
.sources-card p { margin: 0; font-size: 0.85rem; color: #cbd5e1; line-height: 1.4; }
"""

def build_gradio_app():
    with gr.Blocks(title="SECE RAG AI Chatbot") as demo:
        gr.HTML(
            """
            <div class='sece-header'>
                <h1>🎓 Sri Eshwar College of Engineering</h1>
                <p>Advanced RAG Chatbot Assistant (Supporting Groq & Gemini)</p>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3, elem_classes="chatbot-col"):
                gr.Markdown("### 💬 Chat Assistant")
                chatbot = gr.Chatbot(height=450)
                
                with gr.Row():
                    query_input = gr.Textbox(placeholder="Type your question here...", label="Your Query", scale=4)
                    submit_btn = gr.Button("Submit", variant="primary", scale=1, elem_classes="gr-button-primary")
                    
                gr.Markdown("#### 💡 Sample Questions (Click to Ask)")
                with gr.Row():
                    for q_text in SAMPLE_QUESTIONS:
                        gr.Button(q_text, size="sm").click(
                            fn=lambda x=q_text: x,
                            outputs=[query_input]
                        )
            
            with gr.Column(scale=2, elem_classes="sidebar-col"):
                gr.Markdown("### ⚙️ RAG Engine Configuration")
                
                backend_select = gr.Radio(
                    choices=["Groq + Local Embeddings", "Gemini Cloud (Embeddings & LLM)"],
                    value="Groq + Local Embeddings" if engine.mode == "groq" else "Gemini Cloud (Embeddings & LLM)",
                    label="RAG Engine Backend"
                )
                
                api_key_box = gr.Textbox(
                    label="Active API Key",
                    placeholder="Enter Groq Key (gsk_...) or Gemini Key (AIza...)",
                    type="password",
                    value=engine.groq_api_key if engine.mode == "groq" else engine.gemini_api_key
                )
                
                verify_btn = gr.Button("Apply & Verify Config")
                
                # Check status
                if engine.api_configured:
                    initial_status = f"🟢 Configured for {engine.mode.upper()} mode."
                else:
                    initial_status = "🔴 Configure API key to start."
                status_text = gr.Markdown(value=initial_status)
                
                gr.HTML("<hr style='border-color: #334155; margin: 16px 0;' />")
                reindex_btn = gr.Button("Force Rebuild Index (use cache)")
                deep_crawl_btn = gr.Button("🌐 Deep Crawl All Pages & Rebuild", variant="secondary")
                index_status = gr.Markdown("Indices are loaded / dynamically created.")
                
                gr.HTML("<hr style='border-color: #334155; margin: 16px 0;' />")
                gr.Markdown("### 🔍 Retrieval Inspector")
                inspector_panel = gr.HTML("<div style='color: #64748b; font-style: italic;'>Submit a question to see retrieved chunks.</div>")

        def handle_verify(backend, key):
            mode = "groq" if "Groq" in backend else "gemini"
            success, msg = engine.configure_engine(mode, key)
            if success:
                return f"🟢 Configured successfully: {engine.mode.upper()} mode!"
            return f"🔴 {msg}"

        def handle_reindex():
            try:
                chunks = prepare_knowledge_base(use_deep_crawl=True, force_recrawl=False)
                engine.vector_store.build_index(chunks)
                return f"Index rebuilt ({len(chunks)} chunks) for {engine.mode.upper()} mode!"
            except Exception as e:
                return f"Rebuild failed: {e}"

        def handle_deep_crawl():
            """Force re-crawl of ALL sece.ac.in pages and rebuild the index."""
            try:
                yield "Crawling sece.ac.in — this may take 2-5 minutes..."
                chunks = prepare_knowledge_base(use_deep_crawl=True, force_recrawl=True)
                engine.vector_store.build_index(chunks)
                yield f"Deep crawl complete! {len(chunks)} chunks indexed for {engine.mode.upper()} mode."
            except Exception as e:
                yield f"Deep crawl failed: {e}"

        def handle_chat(msg, history):
            if not msg.strip():
                return "", history, ""
            ans, retrieved = engine.generate_answer(msg)
            history.append({"role": "user", "content": msg})
            history.append({"role": "assistant", "content": ans})
            
            inspector_html = ""
            for idx, (text, score) in enumerate(retrieved, 1):
                safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
                inspector_html += f"""
                <div class='sources-card'>
                    <h4>Chunk {idx} — Similarity Score: <b>{score:.4f}</b></h4>
                    <p>{safe_text}</p>
                </div>
                """
            return "", history, inspector_html or "<div style='color: #ef4444;'>No chunks retrieved.</div>"

        def update_key_placeholder(backend):
            if "Groq" in backend:
                return gr.update(value=engine.groq_api_key, placeholder="Enter Groq Key starting with gsk_...")
            else:
                return gr.update(value=engine.gemini_api_key, placeholder="Enter Gemini Key starting with AIza...")

        backend_select.change(
            fn=update_key_placeholder,
            inputs=[backend_select],
            outputs=[api_key_box]
        )

        verify_btn.click(handle_verify, [backend_select, api_key_box], [status_text])
        reindex_btn.click(handle_reindex, outputs=[index_status])
        deep_crawl_btn.click(handle_deep_crawl, outputs=[index_status])
        submit_btn.click(handle_chat, [query_input, chatbot], [query_input, chatbot, inspector_panel])
        query_input.submit(handle_chat, [query_input, chatbot], [query_input, chatbot, inspector_panel])

    return demo


if __name__ == "__main__":
    # Auto configure engine from environment if possible
    engine.auto_init_from_env()

    # Try loading existing FAISS index first; if not found, do a deep crawl build
    if not engine.vector_store.load_index():
        print("No cached FAISS index found. Running initial deep crawl...")
        try:
            chunks = prepare_knowledge_base(use_deep_crawl=True, force_recrawl=False)
            if engine.api_configured:
                engine.vector_store.build_index(chunks)
        except Exception as e:
            print(f"Initial deep crawl failed: {e}. Will build on first API key verify.")

    app = build_gradio_app()
    app.launch(share=False, css=custom_css)
