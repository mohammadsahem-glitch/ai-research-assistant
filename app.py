from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import time
from crewai import Agent, Task, Crew
from crewai_tools import SerpApiGoogleSearchTool
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
    'png', 'jpg', 'jpeg', 'gif', 'bmp'  # Images
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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize database
with app.app_context():
    db.create_all()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Initialize the search tool with API key from environment
# The tool expects SERPAPI_API_KEY environment variable
try:
    # Disable interactive prompts by setting the API key from environment
    serp_api_key = os.environ.get('SERPAPI_API_KEY')
    if not serp_api_key:
        print("Warning: SERPAPI_API_KEY not found in environment variables")
        # Create a dummy tool to prevent crashes
        search_tool = None
    else:
        search_tool = SerpApiGoogleSearchTool()
except Exception as e:
    print(f"Error initializing SerpAPI tool: {e}")
    search_tool = None

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

    elif file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
        # Extract text from images using OCR
        image = Image.open(filepath)
        text = pytesseract.image_to_string(image)
        return text

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

            # Extract text from file
            try:
                content = extract_text_from_file(filepath, filename)
            except Exception as ocr_error:
                # If OCR fails (e.g., tesseract not installed), just note it
                file_ext = filename.rsplit('.', 1)[1].lower()
                if file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
                    content = f"[Image file - OCR not available on this server. Install Tesseract-OCR for text extraction]"
                else:
                    raise ocr_error

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
    return jsonify({
        'status': 'healthy',
        'search_enabled': search_tool is not None,
        'serpapi_configured': bool(os.environ.get('SERPAPI_API_KEY'))
    })

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

        result = crew.kickoff()

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
