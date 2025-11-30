from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import time
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import Field
from typing import Type, Any
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import re
import json
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook
from PIL import Image
import pytesseract
import base64
from openai import OpenAI
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///ai_assistant.db')
# Fix for PostgreSQL URL from some providers
if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {
    'txt', 'md', 'pdf',  # Text and PDF
    'doc', 'docx',  # Word
    'ppt', 'pptx',  # PowerPoint
    'xls', 'xlsx',  # Excel
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'  # Images (with Vision AI support)
}

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    chat_sessions = db.relationship('ChatSession', backref='user', lazy=True, cascade='all, delete-orphan')
    documents = db.relationship('UploadedDocument', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = db.relationship('Message', backref='session', lazy=True, cascade='all, delete-orphan')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user', 'assistant', or 'system'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class UploadedDocument(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(64), unique=True, nullable=False)  # MD5 hash
    content = db.Column(db.Text, nullable=False)
    file_type = db.Column(db.String(20), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

class ExecutionLog(db.Model):
    """Track tool executions for debugging and monitoring"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    tool_name = db.Column(db.String(100), nullable=False)
    tool_type = db.Column(db.String(50), nullable=False)  # 'search', 'vision', 'document', 'agent'
    input_data = db.Column(db.Text, nullable=True)
    output_data = db.Column(db.Text, nullable=True)
    execution_time_ms = db.Column(db.Integer, nullable=True)
    status = db.Column(db.String(20), nullable=False)  # 'success', 'error', 'timeout'
    error_message = db.Column(db.Text, nullable=True)
    extra_info = db.Column(db.Text, nullable=True)  # JSON string for additional info

# Global list for in-memory logs (for tools that run before request context)
execution_logs_memory = []

def log_execution(tool_name, tool_type, input_data, output_data, execution_time_ms, status, error_message=None, extra_info=None, user_id=None):
    """Log a tool execution to the database"""
    try:
        # Truncate long outputs for storage
        input_str = str(input_data)[:5000] if input_data else None
        output_str = str(output_data)[:10000] if output_data else None
        extra_info_str = json.dumps(extra_info) if extra_info else None

        log_entry = ExecutionLog(
            user_id=user_id,
            tool_name=tool_name,
            tool_type=tool_type,
            input_data=input_str,
            output_data=output_str,
            execution_time_ms=execution_time_ms,
            status=status,
            error_message=error_message,
            extra_info=extra_info_str
        )
        db.session.add(log_entry)
        db.session.commit()
    except Exception as e:
        # If DB logging fails, store in memory
        execution_logs_memory.append({
            'timestamp': datetime.utcnow().isoformat(),
            'tool_name': tool_name,
            'tool_type': tool_type,
            'input_data': str(input_data)[:500] if input_data else None,
            'output_data': str(output_data)[:500] if output_data else None,
            'execution_time_ms': execution_time_ms,
            'status': status,
            'error_message': error_message
        })
        print(f"[LOG] Failed to save to DB, stored in memory: {str(e)}")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize database
with app.app_context():
    db.create_all()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Initialize OpenAI client for vision capabilities
openai_client = None
vision_enabled = False
vision_error = None

try:
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
        vision_enabled = True
        print("[SUCCESS] ✓ OpenAI Vision capability initialized")
    else:
        print("[WARNING] OPENAI_API_KEY not found - Vision capability disabled")
        vision_error = "OPENAI_API_KEY not configured"
except Exception as e:
    print(f"[ERROR] Failed to initialize OpenAI client: {str(e)}")
    vision_error = str(e)

def analyze_image_with_vision(filepath, filename, prompt=None):
    """
    Analyze an image using OpenAI's GPT-4 Vision API.
    Returns a detailed description of the image content.
    """
    start_time = time.time()

    if not openai_client:
        # Fallback to OCR if vision is not available
        try:
            image = Image.open(filepath)
            ocr_text = pytesseract.image_to_string(image)
            execution_time = int((time.time() - start_time) * 1000)
            if ocr_text.strip():
                result = f"[OCR Text Extracted - Vision not available]\n{ocr_text}"
                log_execution("OCR (Tesseract)", "vision", filename, result[:500], execution_time, "success",
                             extra_info={"fallback": True, "reason": "Vision API not configured"})
                return result
            log_execution("OCR (Tesseract)", "vision", filename, "No text detected", execution_time, "success",
                         extra_info={"fallback": True, "reason": "Vision API not configured"})
            return "[Image file - No text detected. Vision API not configured for image analysis]"
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            log_execution("OCR (Tesseract)", "vision", filename, None, execution_time, "error", str(e))
            return f"[Image file - Could not process: {str(e)}]"

    try:
        # Read and encode the image
        with open(filepath, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Determine the image type
        file_ext = filename.rsplit('.', 1)[1].lower()
        mime_types = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'webp': 'image/webp'
        }
        mime_type = mime_types.get(file_ext, 'image/jpeg')

        # Default prompt for image analysis
        if not prompt:
            prompt = """Analyze this image in detail. Provide:
1. A comprehensive description of what you see
2. Any text visible in the image (transcribe it)
3. Key objects, people, or elements
4. Colors, layout, and composition
5. Any relevant context or meaning

Be thorough but concise."""

        # Call OpenAI Vision API
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency, supports vision
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500
        )

        vision_analysis = response.choices[0].message.content
        execution_time = int((time.time() - start_time) * 1000)

        # Also try OCR for any text that might be in the image
        try:
            image = Image.open(filepath)
            ocr_text = pytesseract.image_to_string(image)
            if ocr_text.strip():
                vision_analysis += f"\n\n[OCR Extracted Text]:\n{ocr_text.strip()}"
        except:
            pass  # OCR is optional, don't fail if it doesn't work

        # Log successful Vision API execution
        log_execution("GPT-4 Vision", "vision", filename, vision_analysis[:500], execution_time, "success",
                     extra_info={"model": "gpt-4o-mini", "mime_type": mime_type})

        return vision_analysis

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        # Fallback to OCR on error
        try:
            image = Image.open(filepath)
            ocr_text = pytesseract.image_to_string(image)
            if ocr_text.strip():
                result = f"[Vision API error - OCR fallback]\n{ocr_text}"
                log_execution("OCR (Tesseract)", "vision", filename, result[:500], execution_time, "success",
                             extra_info={"fallback": True, "vision_error": str(e)})
                return result
            log_execution("GPT-4 Vision", "vision", filename, None, execution_time, "error", str(e))
            return f"[Image analysis failed: {str(e)}]"
        except Exception as ocr_error:
            log_execution("GPT-4 Vision", "vision", filename, None, execution_time, "error",
                         f"Vision: {str(e)}, OCR: {str(ocr_error)}")
            return f"[Image file - Could not analyze: {str(e)}]"

# Perplexica Search Tool - Open source AI-powered search engine
# GitHub: https://github.com/ItzCrazyKns/Perplexica

class SearchInput(BaseModel):
    """Input schema for the search tool."""
    query: str = Field(description="The search query to look up")

class PerplexicaSearchTool(BaseTool):
    """Search tool that uses Perplexica - an open source AI-powered search engine."""
    name: str = "Web Search"
    description: str = "Search the web for current information, news, and facts using AI-powered search. Use this when you need up-to-date information about any topic."
    args_schema: Type[BaseModel] = SearchInput
    backend_url: str = ""
    chat_model_provider: str = "openai"
    chat_model_name: str = "gpt-4o-mini"
    embedding_model_provider: str = "openai"
    embedding_model_name: str = "text-embedding-3-small"

    def __init__(self, backend_url: str = "", chat_model_provider: str = "openai",
                 chat_model_name: str = "gpt-4o-mini", embedding_model_provider: str = "openai",
                 embedding_model_name: str = "text-embedding-3-small", **kwargs):
        super().__init__(**kwargs)
        self.backend_url = backend_url
        self.chat_model_provider = chat_model_provider
        self.chat_model_name = chat_model_name
        self.embedding_model_provider = embedding_model_provider
        self.embedding_model_name = embedding_model_name

    def _get_focus_mode(self, query: str) -> str:
        """Determine the best focus mode based on the query."""
        query_lower = query.lower()

        # News-related queries
        if any(kw in query_lower for kw in ['news', 'latest', 'recent', 'today', 'breaking', 'update', 'happening']):
            return "webSearch"

        # Academic/research queries
        if any(kw in query_lower for kw in ['research', 'study', 'paper', 'scientific', 'academic', 'journal']):
            return "academicSearch"

        # Video queries
        if any(kw in query_lower for kw in ['video', 'youtube', 'watch', 'tutorial video']):
            return "youtubeSearch"

        # Reddit queries
        if any(kw in query_lower for kw in ['reddit', 'discussion', 'opinions', 'community']):
            return "redditSearch"

        # Default to web search
        return "webSearch"

    def _run(self, query: str) -> str:
        """Execute the search query using Perplexica."""
        start_time = time.time()
        focus_mode = None
        try:
            if not self.backend_url:
                log_execution("Perplexica", "search", query, None, 0, "error", "Backend URL not configured")
                return "Error: Perplexica backend URL not configured"

            # Determine focus mode based on query
            focus_mode = self._get_focus_mode(query)

            # Add current date context for news/recent queries to get up-to-date results
            search_query = query
            query_lower = query.lower()
            current_date = datetime.now()
            current_year = current_date.year
            current_month = current_date.strftime("%B %Y")

            # Check if query is asking for recent/current news
            news_keywords = ['news', 'latest', 'recent', 'today', 'current', 'now', 'update', 'happening', 'this week', 'this month']
            if any(kw in query_lower for kw in news_keywords):
                # Append current date context if not already present
                if str(current_year) not in query and current_date.strftime("%B") not in query:
                    search_query = f"{query} {current_month}"

            # Build request payload
            payload = {
                "chatModel": {
                    "provider": self.chat_model_provider,
                    "model": self.chat_model_name
                },
                "embeddingModel": {
                    "provider": self.embedding_model_provider,
                    "model": self.embedding_model_name
                },
                "focusMode": focus_mode,
                "query": search_query,
                "history": []
            }

            # Make request to Perplexica API
            response = requests.post(
                f"{self.backend_url}/api/search",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )

            execution_time = int((time.time() - start_time) * 1000)

            if response.status_code != 200:
                error_msg = f"Perplexica returned status {response.status_code}"
                log_execution("Perplexica", "search", query, None, execution_time, "error", error_msg,
                             extra_info={"focus_mode": focus_mode, "status_code": response.status_code})
                return f"Search error: {error_msg}"

            result = response.json()

            # Format the response
            formatted_output = []

            # Get the main answer/message
            if "message" in result:
                formatted_output.append(f"**Answer:**\n{result['message']}\n")

            # Get sources/citations if available
            sources_count = 0
            if "sources" in result and result["sources"]:
                sources_count = len(result["sources"])
                formatted_output.append("\n**Sources:**")
                for i, source in enumerate(result["sources"][:5], 1):
                    title = source.get("title", "Unknown")
                    url = source.get("url", "")
                    formatted_output.append(f"{i}. [{title}]({url})")

            output = "\n".join(formatted_output) if formatted_output else f"No results found for: {query}"

            # Log successful execution
            log_execution("Perplexica", "search", query, output[:500], execution_time, "success",
                         extra_info={"focus_mode": focus_mode, "sources_count": sources_count})

            return output

        except requests.exceptions.Timeout:
            execution_time = int((time.time() - start_time) * 1000)
            log_execution("Perplexica", "search", query, None, execution_time, "timeout",
                         "Request timed out", extra_info={"focus_mode": focus_mode})
            return "Search error: Request timed out. Perplexica server may be slow or unavailable."
        except requests.exceptions.ConnectionError:
            execution_time = int((time.time() - start_time) * 1000)
            log_execution("Perplexica", "search", query, None, execution_time, "error",
                         "Connection error", extra_info={"focus_mode": focus_mode})
            return "Search error: Could not connect to Perplexica. Make sure it's running."
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            log_execution("Perplexica", "search", query, None, execution_time, "error",
                         str(e), extra_info={"focus_mode": focus_mode})
            return f"Search error: {str(e)}"

# Initialize search tool - tries Perplexica first, falls back to SerpAPI
search_tool = None
search_tool_error = None
search_tool_type = None

# Try Perplexica first
perplexica_url = os.environ.get('PERPLEXICA_URL', '').strip().strip('"').strip("'")
if perplexica_url:
    print(f"[STARTUP] Initializing Perplexica search tool...")
    print(f"[INFO] Perplexica URL: {perplexica_url}")

    # Get model configuration from environment
    chat_provider = os.environ.get('PERPLEXICA_CHAT_PROVIDER', 'openai')
    chat_model = os.environ.get('PERPLEXICA_CHAT_MODEL', 'gpt-4o-mini')
    embed_provider = os.environ.get('PERPLEXICA_EMBED_PROVIDER', 'openai')
    embed_model = os.environ.get('PERPLEXICA_EMBED_MODEL', 'text-embedding-3-small')

    try:
        # Test connection to Perplexica
        test_response = requests.get(f"{perplexica_url}/api", timeout=10)
        if test_response.status_code == 200:
            search_tool = PerplexicaSearchTool(
                backend_url=perplexica_url,
                chat_model_provider=chat_provider,
                chat_model_name=chat_model,
                embedding_model_provider=embed_provider,
                embedding_model_name=embed_model
            )
            search_tool_type = "Perplexica"
            print(f"[SUCCESS] ✓ Perplexica search tool initialized")
            print(f"[INFO] Chat model: {chat_provider}/{chat_model}")
            print(f"[INFO] Embedding model: {embed_provider}/{embed_model}")
        else:
            print(f"[WARNING] Perplexica returned status {test_response.status_code}")
            search_tool_error = f"Perplexica returned status {test_response.status_code}"
    except requests.exceptions.ConnectionError:
        print(f"[WARNING] Could not connect to Perplexica at {perplexica_url}")
        search_tool_error = "Could not connect to Perplexica"
    except Exception as e:
        print(f"[WARNING] Perplexica initialization failed: {str(e)}")
        search_tool_error = str(e)

# Fall back to SerpAPI if Perplexica not available
if not search_tool:
    serp_api_key = os.environ.get('SERPAPI_API_KEY', '').strip().strip('"').strip("'")
    if serp_api_key:
        print(f"[STARTUP] Falling back to SerpAPI search tool...")
        try:
            from serpapi import GoogleSearch

            # Test the API key
            test_search = GoogleSearch({"q": "test", "api_key": serp_api_key, "num": 1})
            test_results = test_search.get_dict()

            if "error" not in test_results:
                # Create a simple SerpAPI tool as fallback
                class SerpAPIFallbackTool(BaseTool):
                    name: str = "Google Search"
                    description: str = "Search Google for information"
                    args_schema: Type[BaseModel] = SearchInput
                    api_key: str = ""

                    def __init__(self, api_key: str = "", **kwargs):
                        super().__init__(**kwargs)
                        self.api_key = api_key

                    def _run(self, query: str) -> str:
                        start_time = time.time()
                        try:
                            from serpapi import GoogleSearch

                            # Add current date for news queries
                            search_query = query
                            query_lower = query.lower()
                            current_date = datetime.now()
                            current_year = current_date.year
                            current_month = current_date.strftime("%B %Y")

                            search_params = {"q": query, "api_key": self.api_key, "num": 10}

                            # Check if query is asking for recent/current news
                            news_keywords = ['news', 'latest', 'recent', 'today', 'current', 'now', 'update', 'happening']
                            if any(kw in query_lower for kw in news_keywords):
                                # Add date to query and use time-based filter (past month)
                                if str(current_year) not in query:
                                    search_params["q"] = f"{query} {current_month}"
                                search_params["tbs"] = "qdr:m"  # Filter to past month

                            search = GoogleSearch(search_params)
                            results = search.get_dict()
                            execution_time = int((time.time() - start_time) * 1000)

                            output = []
                            results_count = 0
                            if "organic_results" in results:
                                results_count = len(results["organic_results"])
                                for i, r in enumerate(results["organic_results"][:5], 1):
                                    output.append(f"{i}. **{r.get('title', 'No title')}**\n   {r.get('snippet', '')}\n   🔗 {r.get('link', '')}")

                            result_text = "\n".join(output) if output else f"No results for: {query}"

                            # Log successful execution
                            log_execution("SerpAPI", "search", query, result_text[:500], execution_time, "success",
                                         extra_info={"results_count": results_count})

                            return result_text
                        except Exception as e:
                            execution_time = int((time.time() - start_time) * 1000)
                            log_execution("SerpAPI", "search", query, None, execution_time, "error", str(e))
                            return f"Search error: {str(e)}"

                search_tool = SerpAPIFallbackTool(api_key=serp_api_key)
                search_tool_type = "SerpAPI"
                search_tool_error = None
                print(f"[SUCCESS] ✓ SerpAPI fallback initialized")
            else:
                search_tool_error = f"SerpAPI error: {test_results.get('error')}"
                print(f"[ERROR] {search_tool_error}")
        except Exception as e:
            if not search_tool_error:
                search_tool_error = f"SerpAPI failed: {str(e)}"
            print(f"[ERROR] {search_tool_error}")

if not search_tool:
    print(f"[WARNING] No search tool available. Set PERPLEXICA_URL or SERPAPI_API_KEY")

# Define the main conversational agent with search capabilities
# Only include search_tool if it was successfully initialized
agent_tools = [search_tool] if search_tool else []

conversational_agent = Agent(
    role="Intelligent AI Assistant",
    goal="""You are a versatile AI assistant similar to ChatGPT. Your goals are to:
    1. Engage in natural, helpful conversations
    2. Understand user intent and respond appropriately
    3. Use web search when current information is needed
    4. Reference uploaded documents when relevant
    5. Maintain conversation context and remember previous messages
    6. Ask for clarification when the user's request is ambiguous or unclear
    7. Be helpful, accurate, and conversational""",
    backstory="""You are an advanced AI assistant with multiple capabilities:
    - Natural conversation and general knowledge
    - Real-time web search for current information
    - Document analysis and retrieval
    - Context-aware responses that build on previous messages
    You intelligently decide when to search the web, when to use uploaded documents,
    and when to simply converse based on your knowledge. When uncertain about what
    the user wants, you ask clarifying questions.""",
    verbose=False,
    tools=agent_tools,
    allow_delegation=False
)

def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def format_response(text):
    """Format the response with better readability, convert URLs to hyperlinks, detect news widgets, and render tables"""

    # Check if response contains news articles (detect patterns)
    if "```news" in text or "```json" in text:
        # Extract JSON data for news widgets
        json_match = re.search(r'```(?:news|json)\s*(\[.*?\])\s*```', text, re.DOTALL)
        if json_match:
            try:
                news_data = json.loads(json_match.group(1))
                # Create news widget HTML
                news_html = '<div class="news-widgets">'
                for article in news_data[:6]:  # Max 6 articles
                    image = article.get('image', '')
                    title = article.get('title', 'No title')
                    description = article.get('description', '')
                    url = article.get('url', '#')
                    source = article.get('source', 'Unknown')

                    news_html += f'''
                    <div class="news-card">
                        {f'<img src="{image}" alt="{title}" class="news-image">' if image else '<div class="news-image-placeholder">📰</div>'}
                        <div class="news-content">
                            <h4 class="news-title">{title}</h4>
                            <p class="news-description">{description[:150]}...</p>
                            <div class="news-footer">
                                <span class="news-source">📌 {source}</span>
                                <a href="{url}" target="_blank" class="news-link">Read more →</a>
                            </div>
                        </div>
                    </div>
                    '''
                news_html += '</div>'
                # Remove the JSON from text and add widget
                text = re.sub(r'```(?:news|json)\s*\[.*?\]\s*```', news_html, text, flags=re.DOTALL)
            except:
                pass

    # Convert markdown tables to HTML tables
    def convert_markdown_table(match):
        table_text = match.group(0)
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]

        if len(lines) < 2:
            return table_text

        # Parse header
        headers = [cell.strip() for cell in lines[0].split('|') if cell.strip()]

        # Skip separator line (line with dashes)
        data_lines = lines[2:] if len(lines) > 2 else []

        # Build HTML table
        html = '<div class="table-wrapper"><table class="formatted-table">'
        html += '<thead><tr>'
        for header in headers:
            html += f'<th>{header}</th>'
        html += '</tr></thead>'

        html += '<tbody>'
        for line in data_lines:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            html += '<tr>'
            for cell in cells:
                html += f'<td>{cell}</td>'
            html += '</tr>'
        html += '</tbody></table></div>'

        return html

    # Match markdown tables (lines with | separators)
    table_pattern = r'(?:^|\n)(\|.+\|(?:\n\|.+\|)+)(?:\n|$)'
    text = re.sub(table_pattern, convert_markdown_table, text, flags=re.MULTILINE)

    # Convert URLs to HTML links
    url_pattern = r'(https?://[^\s<>"{}|\\^`\[\]]+)'
    text = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)

    # Convert **bold** to HTML bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Convert *italic* to HTML italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # Convert line breaks to HTML breaks
    text = text.replace('\n\n', '<br><br>')
    text = text.replace('\n', '<br>')

    # Format numbered lists
    text = re.sub(r'^(\d+\.)', r'<br>\1', text, flags=re.MULTILINE)

    # Format bullet points
    text = re.sub(r'^- ', r'<br>• ', text, flags=re.MULTILINE)

    return text

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            return jsonify({'success': True, 'message': 'Login successful!'})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'}), 400

        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return jsonify({'success': True, 'message': 'Registration successful!'})

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/architecture')
@login_required
def architecture():
    return render_template('architecture.html')

@app.route('/logs')
@login_required
def logs():
    return render_template('logs.html')

@app.route('/api/logs')
@login_required
def get_logs():
    """API endpoint to fetch execution logs"""
    try:
        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        tool_type = request.args.get('type', None)
        status = request.args.get('status', None)

        # Build query
        query = ExecutionLog.query.order_by(ExecutionLog.timestamp.desc())

        if tool_type:
            query = query.filter(ExecutionLog.tool_type == tool_type)
        if status:
            query = query.filter(ExecutionLog.status == status)

        # Get total count
        total = query.count()

        # Apply pagination
        logs = query.offset(offset).limit(limit).all()

        # Format logs for JSON response
        logs_data = []
        for log in logs:
            extra_info = None
            if log.extra_info:
                try:
                    extra_info = json.loads(log.extra_info)
                except:
                    extra_info = log.extra_info

            logs_data.append({
                'id': log.id,
                'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'tool_name': log.tool_name,
                'tool_type': log.tool_type,
                'input_data': log.input_data,
                'output_data': log.output_data,
                'execution_time_ms': log.execution_time_ms,
                'status': log.status,
                'error_message': log.error_message,
                'extra_info': extra_info
            })

        # Also include in-memory logs
        memory_logs = [{**log, 'source': 'memory'} for log in execution_logs_memory[-50:]]

        return jsonify({
            'success': True,
            'total': total,
            'logs': logs_data,
            'memory_logs': memory_logs
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logs/stats')
@login_required
def get_log_stats():
    """Get statistics about execution logs"""
    try:
        total_executions = ExecutionLog.query.count()
        successful = ExecutionLog.query.filter_by(status='success').count()
        errors = ExecutionLog.query.filter_by(status='error').count()
        timeouts = ExecutionLog.query.filter_by(status='timeout').count()

        # Average execution time
        from sqlalchemy import func
        avg_time = db.session.query(func.avg(ExecutionLog.execution_time_ms)).scalar() or 0

        # Tool type breakdown
        tool_stats = db.session.query(
            ExecutionLog.tool_type,
            func.count(ExecutionLog.id)
        ).group_by(ExecutionLog.tool_type).all()

        # Tool name breakdown
        tool_name_stats = db.session.query(
            ExecutionLog.tool_name,
            func.count(ExecutionLog.id)
        ).group_by(ExecutionLog.tool_name).all()

        return jsonify({
            'success': True,
            'stats': {
                'total': total_executions,
                'successful': successful,
                'errors': errors,
                'timeouts': timeouts,
                'avg_execution_time_ms': round(avg_time, 2),
                'by_type': {t[0]: t[1] for t in tool_stats},
                'by_tool': {t[0]: t[1] for t in tool_name_stats}
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logs/clear', methods=['POST'])
@login_required
def clear_logs():
    """Clear all execution logs"""
    try:
        ExecutionLog.query.delete()
        db.session.commit()
        execution_logs_memory.clear()
        return jsonify({'success': True, 'message': 'Logs cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def extract_text_from_file(filepath, filename):
    """Extract text from different file types"""
    file_ext = filename.rsplit('.', 1)[1].lower()

    if file_ext == 'pdf':
        # Extract text from PDF
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    elif file_ext in ['docx', 'doc']:
        # Extract text from Word documents
        doc = Document(filepath)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    elif file_ext in ['pptx', 'ppt']:
        # Extract text from PowerPoint
        prs = Presentation(filepath)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text

    elif file_ext in ['xlsx', 'xls']:
        # Extract text from Excel
        wb = load_workbook(filepath)
        text = ""
        for sheet in wb.worksheets:
            text += f"Sheet: {sheet.title}\n"
            for row in sheet.iter_rows(values_only=True):
                row_text = "\t".join([str(cell) if cell is not None else "" for cell in row])
                text += row_text + "\n"
        return text

    elif file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']:
        # Analyze images using Vision AI (with OCR fallback)
        return analyze_image_with_vision(filepath, filename)

    else:
        # Read text files (txt, md)
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        session_id = request.form.get('session_id')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Calculate file hash
            file_hash = calculate_file_hash(filepath)

            # Check if file already exists for this user
            existing_doc = UploadedDocument.query.filter_by(file_hash=file_hash, user_id=current_user.id).first()
            if existing_doc:
                os.remove(filepath)  # Remove duplicate file
                return jsonify({
                    'success': True,
                    'message': f'📋 Document "{existing_doc.filename}" is already uploaded!',
                    'filename': existing_doc.filename,
                    'file_type': existing_doc.file_type,
                    'already_exists': True,
                    'char_count': len(existing_doc.content)
                })

            # Extract text/content from file (uses Vision AI for images)
            try:
                content = extract_text_from_file(filepath, filename)
            except Exception as extract_error:
                # Handle extraction errors gracefully
                file_ext = filename.rsplit('.', 1)[1].lower()
                if file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']:
                    content = f"[Image file - Analysis not available: {str(extract_error)}]"
                else:
                    raise extract_error

            file_type = filename.rsplit('.', 1)[1].lower()

            # Store in database
            new_doc = UploadedDocument(
                user_id=current_user.id,
                filename=filename,
                file_hash=file_hash,
                content=content,
                file_type=file_type
            )
            db.session.add(new_doc)
            db.session.commit()

            # If session_id provided, add system message to chat
            if session_id:
                session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
                if session:
                    system_msg = Message(
                        session_id=session_id,
                        role='system',
                        content=f'📎 Uploaded: {filename} ({file_type.upper()}, {len(content)} characters)'
                    )
                    db.session.add(system_msg)
                    db.session.commit()

            return jsonify({
                'success': True,
                'message': f'✅ File "{filename}" uploaded and indexed successfully!',
                'filename': filename,
                'file_type': file_type,
                'already_exists': False,
                'char_count': len(content)
            })
        else:
            return jsonify({'error': '❌ File type not allowed. Allowed types: txt, md, pdf, doc, docx, ppt, pptx, xls, xlsx, png, jpg, jpeg, gif, bmp'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
@login_required
def get_documents():
    """Get all uploaded documents for current user"""
    try:
        docs = UploadedDocument.query.filter_by(user_id=current_user.id).order_by(UploadedDocument.uploaded_at.desc()).all()
        return jsonify({
            'documents': [{
                'id': doc.id,
                'filename': doc.filename,
                'file_type': doc.file_type,
                'uploaded_at': doc.uploaded_at.strftime('%Y-%m-%d %H:%M')
            } for doc in docs]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sessions', methods=['GET'])
@login_required
def get_sessions():
    """Get all chat sessions for current user"""
    try:
        sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.updated_at.desc()).all()
        return jsonify({
            'sessions': [{
                'id': session.id,
                'title': session.title,
                'created_at': session.created_at.strftime('%Y-%m-%d %H:%M'),
                'updated_at': session.updated_at.strftime('%Y-%m-%d %H:%M'),
                'message_count': len(session.messages)
            } for session in sessions]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/session/new', methods=['POST'])
@login_required
def new_session():
    """Create a new chat session for current user"""
    try:
        new_session = ChatSession(user_id=current_user.id, title="New Chat")
        db.session.add(new_session)
        db.session.commit()

        return jsonify({
            'success': True,
            'session_id': new_session.id,
            'title': new_session.title
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/session/<int:session_id>', methods=['GET'])
@login_required
def get_session(session_id):
    """Get a specific chat session with its messages"""
    try:
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
        messages = Message.query.filter_by(session_id=session_id).order_by(Message.timestamp).all()

        return jsonify({
            'session': {
                'id': session.id,
                'title': session.title,
                'created_at': session.created_at.strftime('%Y-%m-%d %H:%M')
            },
            'messages': [{
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.strftime('%Y-%m-%d %H:%M')
            } for msg in messages]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/session/<int:session_id>/delete', methods=['DELETE'])
@login_required
def delete_session(session_id):
    """Delete a chat session"""
    try:
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
        db.session.delete(session)
        db.session.commit()

        return jsonify({'success': True, 'message': 'Session deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify configuration"""
    perplexica_url = os.environ.get('PERPLEXICA_URL', '')
    serp_key_raw = os.environ.get('SERPAPI_API_KEY', '')

    response = {
        'status': 'healthy',
        'search_enabled': search_tool is not None,
        'search_tool_type': search_tool_type,
        'perplexica_configured': bool(perplexica_url),
        'perplexica_url': perplexica_url if perplexica_url else None,
        'serpapi_configured': bool(serp_key_raw),
        'vision_enabled': vision_enabled,
        'openai_configured': bool(os.environ.get('OPENAI_API_KEY'))
    }
    if search_tool_error:
        response['search_error'] = search_tool_error
    if vision_error:
        response['vision_error'] = vision_error
    return jsonify(response)

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        user_message = request.json.get('message', '')
        use_rag = request.json.get('use_rag', False)
        session_id = request.json.get('session_id')
        agent_mode = request.json.get('mode', 'balanced')  # summarize, balanced, verbose, deep_thinking

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Get or create session
        if not session_id:
            new_session = ChatSession(user_id=current_user.id, title="New Chat")
            db.session.add(new_session)
            db.session.commit()
            session_id = new_session.id

        session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        # Update session title if it's the first message
        if len(session.messages) == 0:
            # Generate title from first message
            title = user_message[:50] + "..." if len(user_message) > 50 else user_message
            session.title = title
            db.session.commit()

        # Add user message to database
        user_msg = Message(session_id=session_id, role='user', content=user_message)
        db.session.add(user_msg)
        db.session.commit()

        # Build conversation context from session history
        messages = Message.query.filter_by(session_id=session_id).order_by(Message.timestamp).all()
        context = ""
        if len(messages) > 1:
            # Include last 10 messages for context (5 exchanges)
            recent_history = messages[-11:-1] if len(messages) > 11 else messages[:-1]
            context = "Previous conversation:\n"
            for msg in recent_history:
                context += f"{msg.role.capitalize()}: {msg.content}\n"
            context += "\n"

        # Build document context if RAG is enabled (filter by current user)
        doc_context = ""
        if use_rag:
            docs = UploadedDocument.query.filter_by(user_id=current_user.id).all()
            if docs:
                doc_context = "\n\nAvailable documents:\n"
                for doc in docs:
                    # Limit document content to prevent token overflow
                    limited_content = doc.content[:2000] + "..." if len(doc.content) > 2000 else doc.content
                    doc_context += f"\n[{doc.filename}]:\n{limited_content}\n"

        # Define mode-specific instructions
        mode_instructions = {
            'summarize': """- Provide concise, brief responses
- Focus on key points only
- Use bullet points when possible
- Keep responses short and to the point""",
            'balanced': """- Provide balanced, comprehensive responses
- Include relevant details without being overly verbose
- Use natural conversation flow
- Be helpful and concise""",
            'verbose': """- Provide detailed, thorough responses
- Explain concepts comprehensively
- Include examples and context
- Break down complex topics
- Use multiple paragraphs for clarity""",
            'deep_thinking': """- Analyze the question deeply from multiple angles
- Consider implications and nuances
- Provide reasoning and thought process
- Explore edge cases and alternatives
- Give comprehensive, well-reasoned responses
- Take time to think through the answer carefully"""
        }

        mode_instruction = mode_instructions.get(agent_mode, mode_instructions['balanced'])

        # Create intelligent task description
        task_description = f"""{context}Current user message: {user_message}
{doc_context}

You are an intelligent conversational AI assistant operating in **{agent_mode.replace('_', ' ').title()} Mode**.

Mode-specific guidelines:
{mode_instruction}

Analyze the user's message and respond appropriately:

1. **Simple Conversation**: If the user is greeting, asking a general question, or having casual conversation, respond naturally without using search tools.

2. **Document Query**: If the user asks about uploaded documents (and documents are available), use the document content to answer their question.

3. **Current Information/News**: If the user asks about recent events, news, current data, or anything requiring up-to-date information, use the search tool. When providing news:
   - If you find news articles with images and URLs, format them as a JSON array in markdown code block like this:
   ```news
   [
     {{"title": "Article Title", "description": "Brief description", "url": "https://...", "source": "Source Name", "image": "https://image-url.jpg"}},
     ...
   ]
   ```

4. **Ambiguous Requests**: If the user's request is unclear, vague, or could mean multiple things, politely ask for clarification instead of guessing. For example:
   - "I'm not sure I understand. Could you clarify what you mean by..."
   - "To help you better, could you provide more details about..."
   - "Are you asking about X or Y? Please let me know so I can assist you better."

5. **Context Awareness**: Remember and reference previous messages in the conversation when relevant.

Guidelines:
- Be conversational and natural with appropriate emojis
- Use search tool ONLY when current/factual information is needed
- Reference documents when the question relates to them
- Maintain context from previous messages
- If you use search, include source URLs
- When uncertain or confused, ASK for clarification
- Be helpful and concise"""

        # Create task
        task = Task(
            description=task_description,
            agent=conversational_agent,
            expected_output="A natural, context-aware response that appropriately addresses the user's message or asks for clarification if needed."
        )

        # Create and execute crew
        crew = Crew(
            agents=[conversational_agent],
            tasks=[task]
        )

        chat_start_time = time.time()
        try:
            result = crew.kickoff()
            chat_execution_time = int((time.time() - chat_start_time) * 1000)

            # Log successful chat execution
            log_execution(
                "CrewAI Agent", "agent",
                user_message[:500],
                str(result)[:1000],
                chat_execution_time,
                "success",
                extra_info={"mode": agent_mode, "use_rag": use_rag, "session_id": session_id}
            )
        except Exception as agent_error:
            chat_execution_time = int((time.time() - chat_start_time) * 1000)
            log_execution(
                "CrewAI Agent", "agent",
                user_message[:500],
                None,
                chat_execution_time,
                "error",
                str(agent_error),
                extra_info={"mode": agent_mode, "use_rag": use_rag}
            )
            raise agent_error

        # Add assistant response to database
        assistant_response = str(result)
        assistant_msg = Message(session_id=session_id, role='assistant', content=assistant_response)
        db.session.add(assistant_msg)

        # Update session timestamp
        session.updated_at = datetime.utcnow()
        db.session.commit()

        # Format the response
        formatted_result = format_response(assistant_response)

        # Determine what actions were taken for progress simulation
        actions_taken = []
        if use_rag and docs:
            actions_taken.append('documents')
        if 'search' in assistant_response.lower() or 'http' in assistant_response.lower():
            actions_taken.append('web_search')
        actions_taken.append('thinking')

        return jsonify({
            'response': formatted_result,
            'session_id': session_id,
            'session_title': session.title,
            'mode': agent_mode,
            'actions': actions_taken
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For Railway deployment, bind to 0.0.0.0 and use PORT from environment
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
