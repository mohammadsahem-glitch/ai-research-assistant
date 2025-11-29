# AI Research Assistant with User Authentication

A powerful AI research assistant that combines web search capabilities with document analysis, now with secure user authentication and multi-user support.

## Features

### 1. User Authentication & Security 🔐
- Secure login and registration system
- Password hashing with Werkzeug
- Session management with Flask-Login
- User-specific chat history and documents
- Logout functionality

### 2. Web Search Research 🔍
- Powered by SerpAPI for real-time web searches
- Comprehensive summaries with source citations
- Clickable hyperlinks to sources
- News widgets with images (ChatGPT-style)

### 3. Document Upload & Analysis 📚
- Upload documents (.txt, .md, .pdf, .doc, .docx, .ppt, .pptx, .xls, .xlsx, images)
- Documents are stored per user (private to each account)
- Ask questions about your uploaded documents
- Combine document knowledge with web search
- Duplicate file detection with MD5 hashing
- OCR support for images

### 4. Beautiful Chat Interface 💬
- Modern gradient design with ChatGPT-style UI
- Sidebar with chat sessions
- Multiple agent modes (Summarize, Balanced, Verbose, Deep Thinking)
- Interactive progress indicators
- Formatted responses with tables, bold, italic, and links
- News widgets with images
- Mobile responsive

## How to Use

### Start the Server

**Option 1**: Double-click [start_server.bat](start_server.bat)

**Option 2**: Run from command line:
```bash
python app.py
```

Then open your browser to: **http://127.0.0.1:5000**

### First Time Setup

1. When you first access the app, you'll see the login page
2. Click "Register here" to create a new account
3. Enter your email and password (minimum 6 characters)
4. You'll be automatically logged in and redirected to the chat interface

### Using the App

1. **New Chat**: Click "➕ New Chat" to start a new conversation
2. **Switch Chats**: Click on previous chat sessions in the sidebar
3. **Delete Chats**: Hover over a chat and click the delete button (🗑️)

### Upload Documents

1. Click "Upload Document" button
2. Select a file (.txt, .md, .pdf, .doc, or .docx)
3. Wait for indexing to complete
4. Check the "Use uploaded documents" checkbox

### Ask Questions

**Without documents**: The AI will search the web and provide answers with sources

**With documents enabled**: The AI will:
- Search your uploaded documents first
- Combine document knowledge with web search
- Provide comprehensive answers from both sources

## File Structure

```
my_ai_agent/
├── app.py                          # Main Flask application with authentication
├── .env                            # API keys (OpenAI, SerpAPI, SECRET_KEY)
├── start_server.bat                # Easy startup script
├── requirements.txt                # Python dependencies
├── Procfile                        # Deployment configuration
├── runtime.txt                     # Python version for deployment
├── .gitignore                      # Git ignore file
├── DEPLOYMENT.md                   # Deployment guide
├── templates/
│   ├── index.html                  # Chat interface
│   ├── login.html                  # Login page
│   └── register.html               # Registration page
├── uploads/                        # Uploaded files storage
├── ai_assistant.db                 # SQLite database (users, chats, documents)
└── README.md                       # This file
```

## Configuration

### API Keys Required

Edit `.env` file:

```env
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_key_here

# SerpAPI Key (required for web search)
SERPAPI_API_KEY=your_serpapi_key_here
```

### Supported File Types

- Text files: `.txt`, `.md`
- Documents: `.pdf`, `.doc`, `.docx`
- Maximum file size: 16 MB

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Authentication**: Flask-Login with Werkzeug password hashing
- **Database**: SQLite (SQLAlchemy ORM) - PostgreSQL ready
- **AI Agent**: CrewAI with OpenAI GPT-4o-mini
- **Web Search**: SerpAPI
- **Document Processing**: PyPDF2, python-docx, python-pptx, openpyxl, Pillow + Tesseract OCR
- **Frontend**: Vanilla JavaScript with modern CSS
- **Deployment**: Gunicorn WSGI server

## Dependencies

All dependencies are listed in [requirements.txt](requirements.txt). Install with:

```bash
pip install -r requirements.txt
```

Key dependencies:
- flask, flask-sqlalchemy, flask-login
- crewai, crewai-tools
- PyPDF2, python-docx, python-pptx, openpyxl, pillow, pytesseract
- serpapi, google-search-results
- gunicorn (for production deployment)

## How Document Processing Works

1. **Upload**: Document is uploaded and saved to `uploads/`
2. **Hash Calculation**: MD5 hash generated to detect duplicates
3. **Text Extraction**: Content extracted based on file type (PDF, Word, Excel, etc.)
4. **Storage**: Text stored in SQLite database, linked to user account
5. **Query**: When chat uses documents, relevant content is retrieved from database
6. **Generation**: AI generates answer using document content + web search

## Tips

- Each user has their own private documents and chat history
- Upload multiple documents for a richer knowledge base
- Use specific questions for better document retrieval
- Disable "Use uploaded documents" for pure web search
- Try different agent modes for different use cases:
  - **Summarize**: Quick, concise answers
  - **Balanced**: Good mix of detail and brevity
  - **Verbose**: Detailed explanations
  - **Deep Thinking**: Analytical, multi-perspective responses
- Documents and chats persist across server restarts
- Use Logout button to securely end your session

## Deployment to Internet

See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive deployment guide covering:
- Railway (recommended - easiest)
- Render
- Heroku
- PythonAnywhere

All with step-by-step instructions for publishing your app online with HTTPS!

## Troubleshooting

**Can't access login page**: Server may not have started - check console for errors

**Server won't start**: Make sure port 5000 is not in use

**File upload fails**: Check file size (<50MB) and file type

**No API responses**: Verify API keys in `.env` file

**Documents not being used**: Make sure "Use uploaded documents" is checked

**Login not working**: Delete `ai_assistant.db` and restart to recreate database

**Forgot password**: Currently no reset feature - contact admin or recreate database

## Security Notes

- Passwords are hashed using Werkzeug (industry standard)
- Session cookies are httponly and secure
- Each user can only access their own data
- API keys stored in environment variables (never in code)
- For production: Use PostgreSQL instead of SQLite
- For production: Set strong SECRET_KEY environment variable

## License

MIT License
