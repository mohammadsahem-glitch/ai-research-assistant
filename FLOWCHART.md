# AI Research Assistant - System Flowchart

## 🏗️ Application Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE (Browser)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  login.html  │  │register.html │  │  index.html  │                  │
│  │              │  │              │  │ (Main Chat)  │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
└─────────┼──────────────────┼──────────────────┼──────────────────────────┘
          │                  │                  │
          │ AJAX/JSON        │ AJAX/JSON        │ AJAX/JSON
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FLASK APPLICATION (app.py)                        │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    AUTHENTICATION ROUTES                       │    │
│  │  /login (GET, POST)  │  /register (GET, POST)  │  /logout     │    │
│  │         │                     │                       │         │    │
│  │         └─────────────────────┴───────────────────────┘         │    │
│  │                              │                                  │    │
│  │                    Flask-Login (UserMixin)                      │    │
│  └────────────────────────────────┬───────────────────────────────┘    │
│                                   │                                     │
│  ┌────────────────────────────────┴───────────────────────────────┐    │
│  │                    PROTECTED ROUTES (@login_required)          │    │
│  │                                                                 │    │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐ │    │
│  │  │  CHAT ROUTES    │  │  SESSION ROUTES  │  │  DOC ROUTES   │ │    │
│  │  │                 │  │                  │  │               │ │    │
│  │  │  /chat (POST)   │  │  /sessions (GET) │  │  /upload      │ │    │
│  │  │       │         │  │  /session/new    │  │  /documents   │ │    │
│  │  │       │         │  │  /session/:id    │  │               │ │    │
│  │  │       │         │  │  /session/delete │  │               │ │    │
│  │  └───────┼─────────┘  └──────────────────┘  └───────────────┘ │    │
│  │          │                                                     │    │
│  └──────────┼─────────────────────────────────────────────────────┘    │
│             │                                                           │
│             ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CORE AI PROCESSING ENGINE                    │   │
│  │                                                                 │   │
│  │  1. Parse user message                                         │   │
│  │  2. Load conversation context (last 10 messages)               │   │
│  │  3. Load user documents (if RAG enabled)                       │   │
│  │  4. Apply agent mode (summarize/balanced/verbose/deep)         │   │
│  │  5. Execute CrewAI agent with task                             │   │
│  │  6. Format response (tables, news, links, markdown)            │   │
│  │  7. Save to database                                           │   │
│  └──────────────────────┬──────────────────────────────────────────┘   │
│                         │                                              │
│         ┌───────────────┼───────────────┐                              │
│         │               │               │                              │
│         ▼               ▼               ▼                              │
│  ┌──────────┐  ┌────────────────┐  ┌─────────────┐                    │
│  │  CrewAI  │  │  Document      │  │  Response   │                    │
│  │  Agent   │  │  Processing    │  │  Formatter  │                    │
│  │          │  │                │  │             │                    │
│  │ GPT-4o   │  │ • PDF Extract  │  │ • Tables    │                    │
│  │ -mini    │  │ • Word Extract │  │ • News      │                    │
│  │          │  │ • Excel Parse  │  │ • Markdown  │                    │
│  │ SerpAPI  │  │ • PPT Extract  │  │ • Links     │                    │
│  │ Search   │  │ • OCR Images   │  │             │                    │
│  └──────────┘  └────────────────┘  └─────────────┘                    │
│                                                                         │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATABASE LAYER (SQLAlchemy + SQLite)                 │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │    User      │  │ ChatSession  │  │   Message    │  │  Uploaded  │ │
│  │              │  │              │  │              │  │  Document  │ │
│  │ • id         │  │ • id         │  │ • id         │  │ • id       │ │
│  │ • email      │  │ • user_id ─┐ │  │ • session_id │  │ • user_id ─┼─┐
│  │ • password   │  │ • title    │ │  │ • role       │  │ • filename │ │
│  │ • created_at │  │ • created  │ │  │ • content    │  │ • hash     │ │
│  └──────┬───────┘  │ • updated  │ │  │ • timestamp  │  │ • content  │ │
│         │          └────────────┘ │  └──────────────┘  │ • type     │ │
│         │                 │       │                    └────────────┘ │
│         │                 │       │                           │       │
│         └─────────────────┴───────┴───────────────────────────┘       │
│                      (One-to-Many Relationships)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Detailed Flow Diagrams

### 1. User Authentication Flow

```
START
  ├─→ User visits app
  │
  ├─→ Is user authenticated?
  │   ├─ YES → Redirect to /home (index.html)
  │   │
  │   └─ NO → Show /login page
  │       │
  │       ├─→ User clicks "Register"
  │       │   ├─→ Show /register page
  │       │   ├─→ User submits email + password
  │       │   ├─→ Check if email exists
  │       │   │   ├─ YES → Return error "Email already registered"
  │       │   │   └─ NO → Continue
  │       │   ├─→ Hash password (Werkzeug)
  │       │   ├─→ Create User record in database
  │       │   ├─→ Auto-login user (Flask-Login)
  │       │   └─→ Redirect to /home
  │       │
  │       └─→ User submits login
  │           ├─→ Validate email + password
  │           │   ├─ VALID → Login user (Flask-Login)
  │           │   │           └─→ Redirect to /home
  │           │   │
  │           │   └─ INVALID → Return error "Invalid credentials"
  │           │
  │           └─→ Continue session with cookie
  │
  └─→ User clicks Logout
      └─→ Call /logout endpoint
          └─→ Clear session
              └─→ Redirect to /login
```

---

### 2. Chat Message Processing Flow

```
User sends message from index.html
  │
  ├─→ JavaScript: POST /chat with JSON:
  │   {
  │     "message": "user input",
  │     "use_rag": true/false,
  │     "session_id": 123,
  │     "mode": "balanced"
  │   }
  │
  ▼
Flask /chat route (@login_required)
  │
  ├─→ [1] VALIDATE INPUT
  │   └─→ Check message is not empty
  │
  ├─→ [2] GET OR CREATE SESSION
  │   ├─ Session ID exists? → Load from database
  │   └─ No session? → Create new ChatSession
  │
  ├─→ [3] UPDATE SESSION TITLE
  │   └─ If first message → Use first 50 chars as title
  │
  ├─→ [4] SAVE USER MESSAGE
  │   └─→ Create Message record (role='user')
  │
  ├─→ [5] BUILD CONTEXT
  │   │
  │   ├─→ Load conversation history
  │   │   └─→ Get last 10 messages from session
  │   │
  │   ├─→ Load documents (if use_rag=true)
  │   │   └─→ Query UploadedDocument table (user_id=current_user)
  │   │       └─→ Limit each doc to 2000 chars
  │   │
  │   └─→ Apply agent mode settings
  │       ├─ "summarize" → Concise responses, max_iter=3
  │       ├─ "balanced" → Standard responses, max_iter=5
  │       ├─ "verbose" → Detailed responses, max_iter=8
  │       └─ "deep_thinking" → Thorough analysis, max_iter=12
  │
  ├─→ [6] CREATE CREWAI AGENT
  │   │
  │   ├─→ Initialize Agent:
  │   │   ├─ role: "Intelligent Research Assistant"
  │   │   ├─ goal: Based on mode + context
  │   │   ├─ backstory: Context-aware assistant description
  │   │   ├─ tools: [SerpApiGoogleSearchTool] (web search)
  │   │   ├─ llm: gpt-4o-mini
  │   │   └─ verbose: Based on mode
  │   │
  │   ├─→ Create Task:
  │   │   ├─ description: Full context + user message + instructions
  │   │   ├─ expected_output: Formatted response with news/tables
  │   │   └─ agent: The agent created above
  │   │
  │   ├─→ Create Crew:
  │   │   ├─ agents: [agent]
  │   │   ├─ tasks: [task]
  │   │   └─ verbose: Based on mode
  │   │
  │   └─→ Execute: crew.kickoff()
  │       │
  │       ├─→ Agent analyzes intent:
  │       │   ├─ Simple conversation? → Direct response
  │       │   ├─ Document query? → Use doc_context
  │       │   └─ Current info needed? → Use SerpAPI search
  │       │
  │       └─→ Return AI response
  │
  ├─→ [7] FORMAT RESPONSE
  │   │
  │   ├─→ Detect news widgets
  │   │   └─→ Convert ```news [JSON] ``` → HTML cards
  │   │
  │   ├─→ Convert markdown tables
  │   │   └─→ | Header | → <table> with CSS
  │   │
  │   ├─→ Convert URLs to clickable links
  │   ├─→ Convert **bold** → <strong>
  │   ├─→ Convert *italic* → <em>
  │   └─→ Format bullet points and numbers
  │
  ├─→ [8] SAVE ASSISTANT MESSAGE
  │   └─→ Create Message record (role='assistant')
  │
  ├─→ [9] UPDATE SESSION TIMESTAMP
  │   └─→ Set updated_at = now()
  │
  └─→ [10] RETURN JSON RESPONSE
      └─→ {
            "response": formatted_html,
            "session_id": 123,
            "session_title": "Chat title",
            "mode": "balanced",
            "actions": ["documents", "web_search", "thinking"]
          }
```

---

### 3. Document Upload & Processing Flow

```
User selects file in index.html
  │
  ├─→ JavaScript: POST /upload (multipart/form-data)
  │
  ▼
Flask /upload route (@login_required)
  │
  ├─→ [1] VALIDATE FILE
  │   ├─→ Check file exists in request
  │   ├─→ Check filename not empty
  │   └─→ Check file extension allowed
  │       └─ Allowed: txt, md, pdf, doc, docx, ppt, pptx,
  │                   xls, xlsx, png, jpg, jpeg, gif, bmp
  │
  ├─→ [2] SAVE FILE TEMPORARILY
  │   └─→ Save to uploads/ folder
  │
  ├─→ [3] CALCULATE MD5 HASH
  │   └─→ Hash entire file content
  │
  ├─→ [4] CHECK FOR DUPLICATES
  │   ├─→ Query: UploadedDocument where
  │   │         user_id=current_user AND file_hash=hash
  │   │
  │   ├─ FOUND? → Delete temp file
  │   │           └─→ Return "Already uploaded"
  │   │
  │   └─ NOT FOUND? → Continue
  │
  ├─→ [5] EXTRACT TEXT BASED ON FILE TYPE
  │   │
  │   ├─→ PDF (.pdf)
  │   │   └─→ PyPDF2.PdfReader
  │   │       └─→ Loop through pages
  │   │           └─→ Extract text from each page
  │   │
  │   ├─→ Word (.doc, .docx)
  │   │   └─→ python-docx.Document
  │   │       └─→ Extract paragraphs and tables
  │   │
  │   ├─→ PowerPoint (.ppt, .pptx)
  │   │   └─→ python-pptx.Presentation
  │   │       └─→ Loop through slides
  │   │           └─→ Extract shapes with text
  │   │
  │   ├─→ Excel (.xls, .xlsx)
  │   │   └─→ openpyxl.load_workbook
  │   │       └─→ Loop through sheets
  │   │           └─→ Extract all cell values
  │   │
  │   ├─→ Images (.png, .jpg, .jpeg, .gif, .bmp)
  │   │   └─→ PIL.Image.open
  │   │       └─→ pytesseract.image_to_string (OCR)
  │   │
  │   └─→ Text (.txt, .md)
  │       └─→ Direct file.read()
  │
  ├─→ [6] SAVE TO DATABASE
  │   └─→ Create UploadedDocument:
  │       ├─ user_id: current_user.id
  │       ├─ filename: original name
  │       ├─ file_hash: MD5 hash
  │       ├─ content: extracted text
  │       ├─ file_type: extension
  │       └─ uploaded_at: timestamp
  │
  ├─→ [7] CLEAN UP
  │   └─→ Keep temp file (may need later)
  │       OR delete if configured
  │
  └─→ [8] RETURN SUCCESS
      └─→ {
            "success": true,
            "message": "File uploaded!",
            "filename": "document.pdf"
          }
```

---

### 4. Session Management Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SESSION LIFECYCLE                        │
└─────────────────────────────────────────────────────────────┘

CREATE NEW SESSION
  │
  ├─→ User clicks "New Chat" button
  │   └─→ POST /session/new
  │       ├─→ Create ChatSession(user_id, title="New Chat")
  │       ├─→ Save to database
  │       └─→ Return session_id
  │           └─→ Frontend switches to new session
  │
LOAD EXISTING SESSION
  │
  ├─→ User clicks session in sidebar
  │   └─→ GET /session/:id
  │       ├─→ Load ChatSession by id + user_id
  │       ├─→ Load all Message records for session
  │       └─→ Return {session, messages[]}
  │           └─→ Frontend displays conversation
  │
LIST ALL SESSIONS
  │
  ├─→ Page load / Refresh
  │   └─→ GET /sessions
  │       ├─→ Query ChatSession where user_id=current_user
  │       ├─→ Order by updated_at DESC
  │       └─→ Return list with metadata
  │           └─→ Frontend renders sidebar
  │
DELETE SESSION
  │
  └─→ User clicks delete icon
      └─→ DELETE /session/:id/delete
          ├─→ Find ChatSession by id + user_id
          ├─→ Delete session (cascade deletes messages)
          └─→ Return success
              └─→ Frontend removes from sidebar
```

---

### 5. Agent Mode Decision Tree

```
User selects mode: [Summarize | Balanced | Verbose | Deep Thinking]
  │
  ├─→ SUMMARIZE MODE
  │   ├─ Instructions: "Provide concise, brief responses"
  │   ├─ Max iterations: 3
  │   ├─ Verbose logging: False
  │   ├─ Output style: Bullet points, key facts only
  │   └─ Example: "What is AI?" → "AI: computer systems that mimic human intelligence"
  │
  ├─→ BALANCED MODE (Default)
  │   ├─ Instructions: "Provide balanced, comprehensive responses"
  │   ├─ Max iterations: 5
  │   ├─ Verbose logging: True
  │   ├─ Output style: Natural conversation with relevant details
  │   └─ Example: "What is AI?" → Paragraph with definition, examples, applications
  │
  ├─→ VERBOSE MODE
  │   ├─ Instructions: "Provide detailed, thorough responses"
  │   ├─ Max iterations: 8
  │   ├─ Verbose logging: True
  │   ├─ Output style: Multiple paragraphs, examples, context
  │   └─ Example: "What is AI?" → Full explanation with history, types, use cases
  │
  └─→ DEEP THINKING MODE
      ├─ Instructions: "Analyze deeply from multiple angles"
      ├─ Max iterations: 12
      ├─ Verbose logging: True
      ├─ Output style: Comprehensive analysis with reasoning
      └─ Example: "What is AI?" → Deep dive into philosophy, ethics, technical details
```

---

### 6. Response Formatting Pipeline

```
AI Agent returns raw text response
  │
  ├─→ [STEP 1] Detect & Convert News Widgets
  │   │
  │   ├─→ Search for pattern: ```news [...JSON...] ```
  │   ├─→ Parse JSON array
  │   ├─→ Generate HTML:
  │   │   <div class="news-widget">
  │   │     <div class="news-item">
  │   │       <img src="...">
  │   │       <h4><a href="...">Title</a></h4>
  │   │       <p>Snippet</p>
  │   │       <span>Source</span>
  │   │     </div>
  │   │   </div>
  │   └─→ Replace code block with HTML
  │
  ├─→ [STEP 2] Convert Markdown Tables
  │   │
  │   ├─→ Search for pattern: | Header | Header |
  │   │                        |--------|--------|
  │   │                        | Cell   | Cell   |
  │   │
  │   ├─→ Parse table structure
  │   ├─→ Generate HTML:
  │   │   <table class="formatted-table">
  │   │     <thead><tr><th>...</th></tr></thead>
  │   │     <tbody><tr><td>...</td></tr></tbody>
  │   │   </table>
  │   └─→ Replace markdown with HTML table
  │
  ├─→ [STEP 3] Convert URLs to Links
  │   │
  │   ├─→ Regex: https?://[...]
  │   └─→ Replace: <a href="URL" target="_blank">URL</a>
  │
  ├─→ [STEP 4] Convert Bold Text
  │   │
  │   ├─→ Regex: **text**
  │   └─→ Replace: <strong>text</strong>
  │
  ├─→ [STEP 5] Convert Italic Text
  │   │
  │   ├─→ Regex: *text*
  │   └─→ Replace: <em>text</em>
  │
  ├─→ [STEP 6] Format Lists
  │   │
  │   ├─→ Numbered: 1. Item → <br>1. Item
  │   └─→ Bullets: - Item → <br>• Item
  │
  └─→ [STEP 7] Return Formatted HTML
      └─→ Rendered in chat message div
```

---

### 7. Database Schema Relationships

```
┌──────────────────────────────────────────────────────────────────┐
│                      DATABASE RELATIONSHIPS                       │
└──────────────────────────────────────────────────────────────────┘

User (1) ────────────── (Many) ChatSession
  │                               │
  │                               │
  │                         (Many) Message
  │                               │
  │                          (Cascade Delete:
  │                           Delete user →
  │                           Delete all sessions →
  │                           Delete all messages)
  │
  └─────────────────── (Many) UploadedDocument
                               │
                          (Cascade Delete:
                           Delete user →
                           Delete all documents)

UNIQUE CONSTRAINTS:
  • User.email → UNIQUE
  • UploadedDocument.file_hash → UNIQUE (prevents duplicate uploads)

INDEXES (Auto-created on Foreign Keys):
  • ChatSession.user_id
  • Message.session_id
  • UploadedDocument.user_id
```

---

### 8. Technology Stack Map

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                          │
│  • HTML5 + CSS3 (ChatGPT-style dark theme)                     │
│  • Vanilla JavaScript (Fetch API, DOM manipulation)            │
│  • Responsive layout (sidebar + main chat area)                │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND FRAMEWORK                          │
│  • Flask 3.1.2 (Python web framework)                          │
│  • Flask-SQLAlchemy 3.1.1 (ORM)                                │
│  • Flask-Login 0.6.3 (Authentication)                          │
│  • Werkzeug 3.1.3 (Security, password hashing)                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐
│   AI LAYER   │  │ DOCUMENT LAYER   │  │  DATABASE       │
│              │  │                  │  │                 │
│ • CrewAI     │  │ • PyPDF2 (PDF)   │  │ • SQLite        │
│   0.86.0     │  │ • python-docx    │  │ • PostgreSQL    │
│              │  │   (Word)         │  │   (Production)  │
│ • OpenAI     │  │ • python-pptx    │  │                 │
│   GPT-4o-mini│  │   (PowerPoint)   │  │ • SQLAlchemy    │
│              │  │ • openpyxl       │  │   Models        │
│ • SerpAPI    │  │   (Excel)        │  │                 │
│   (Search)   │  │ • Pillow +       │  │ • Migrations    │
│              │  │   pytesseract    │  │   on deploy     │
│              │  │   (OCR)          │  │                 │
└──────────────┘  └──────────────────┘  └─────────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT LAYER                             │
│  • Gunicorn 23.0.0 (WSGI server)                               │
│  • Railway / Render / Heroku (PaaS)                            │
│  • Python 3.12.8 runtime                                       │
│  • Environment variables (.env → production config)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Features Flow Summary

### Multi-User Support
```
User A                          User B
  │                              │
  ├─→ Login with email A         ├─→ Login with email B
  ├─→ Upload doc1.pdf            ├─→ Upload doc2.pdf
  ├─→ Create session "Chat A"    ├─→ Create session "Chat B"
  ├─→ Ask about doc1.pdf         ├─→ Ask about doc2.pdf
  │   (Can ONLY see doc1.pdf)    │   (Can ONLY see doc2.pdf)
  │                              │
  └─→ Isolated data ✓            └─→ Isolated data ✓
```

### RAG (Retrieval-Augmented Generation)
```
User enables "Use Documents" toggle
  │
  ├─→ Backend loads UploadedDocument.content
  ├─→ Includes in agent context (max 2000 chars/doc)
  ├─→ Agent uses document knowledge to answer
  └─→ Response includes doc-specific information
```

### Conversation Context
```
Message 1: "What is Python?"
  └─→ Save to session

Message 2: "Tell me more about it"
  └─→ Load last 10 messages
  └─→ Agent understands "it" = Python
  └─→ Continues conversation naturally
```

### Duplicate Detection
```
User uploads "report.pdf" (hash: abc123)
  └─→ Save to database

User uploads same file again
  └─→ Calculate hash: abc123
  └─→ Find existing record
  └─→ Reject upload: "Already uploaded"
```

---

## 📁 File Structure

```
my_ai_agent/
│
├── app.py                 # Main Flask application (498 lines)
│   ├── Database models
│   ├── Authentication routes
│   ├── Chat processing
│   ├── Document upload
│   └── Session management
│
├── templates/
│   ├── index.html         # Main chat interface (826 lines)
│   ├── login.html         # Login page
│   └── register.html      # Registration page
│
├── uploads/               # Temporary file storage
│
├── instance/              # SQLite database folder
│   └── ai_assistant.db    # User data, sessions, messages, docs
│
├── .env                   # API keys (not in git)
│   ├── OPENAI_API_KEY
│   └── SERPAPI_API_KEY
│
├── requirements.txt       # Python dependencies
├── Procfile              # Deployment config (Gunicorn)
├── runtime.txt           # Python version (3.12.8)
├── .gitignore            # Git exclusions
├── DEPLOYMENT.md         # Deployment guide (311 lines)
└── FLOWCHART.md          # This file
```

---

## 🔄 End-to-End Example: User Asks Question About News

```
1. USER ACTION
   └─→ Opens index.html
       └─→ Types: "What are the latest AI developments?"
           └─→ Clicks Send

2. FRONTEND
   └─→ JavaScript captures message
       └─→ POST /chat with JSON:
           {
             "message": "What are the latest AI developments?",
             "use_rag": false,
             "session_id": 5,
             "mode": "balanced"
           }

3. BACKEND (/chat route)
   └─→ Validate user is logged in (@login_required)
   └─→ Load session #5 for current user
   └─→ Save user message to database
   └─→ Load last 10 messages for context
   └─→ Build task description:
       "You are in Balanced Mode.
        User asks: What are the latest AI developments?
        Use search tool for current information."

4. CREWAI AGENT
   └─→ Agent analyzes: "This needs current information"
   └─→ Uses SerpApiGoogleSearchTool
   └─→ Searches: "latest AI developments 2025"
   └─→ Finds 5 articles with titles, URLs, images
   └─→ Formats response:
       ```news
       [
         {
           "title": "GPT-5 Released",
           "link": "https://...",
           "image": "https://...",
           "snippet": "...",
           "source": "TechNews"
         },
         ...
       ]
       ```

5. RESPONSE FORMATTER
   └─→ Detects ```news block
   └─→ Converts to HTML:
       <div class="news-widget">
         <div class="news-item">
           <img src="...">
           <h4><a href="...">GPT-5 Released</a></h4>
           <p>...</p>
           <span>TechNews</span>
         </div>
         ...
       </div>

6. DATABASE
   └─→ Save assistant message to database
   └─→ Update session.updated_at

7. RESPONSE
   └─→ Return JSON:
       {
         "response": "<formatted HTML>",
         "session_id": 5,
         "session_title": "What are the latest...",
         "mode": "balanced",
         "actions": ["web_search", "thinking"]
       }

8. FRONTEND
   └─→ Receive JSON response
   └─→ Add bot message to chat
   └─→ Render news widget with images
   └─→ Scroll to bottom
   └─→ Ready for next message
```

---

## 🔐 Security Features

```
1. PASSWORD SECURITY
   ├─→ Werkzeug password hashing (PBKDF2)
   ├─→ Salted hashes stored in database
   └─→ Never store plaintext passwords

2. SESSION SECURITY
   ├─→ Flask session cookies (HTTP-only)
   ├─→ SECRET_KEY from environment
   └─→ @login_required decorator on all protected routes

3. DATA ISOLATION
   ├─→ All queries filter by user_id
   ├─→ Users can ONLY see their own:
   │   ├─ Chat sessions
   │   ├─ Messages
   │   └─ Documents
   └─→ 404 error if accessing other user's data

4. INPUT VALIDATION
   ├─→ File type whitelist
   ├─→ File size limit (50MB)
   ├─→ SQL injection protection (SQLAlchemy ORM)
   └─→ XSS protection (Jinja2 auto-escaping)

5. PRODUCTION SECURITY
   ├─→ HTTPS enforced (Railway/Render/Heroku)
   ├─→ Environment variables for secrets
   ├─→ .gitignore prevents leaking .env
   └─→ PostgreSQL URL sanitization
```

---

## ⚡ Performance Optimizations

```
1. DATABASE QUERIES
   ├─→ Lazy loading relationships
   ├─→ Index on foreign keys (auto)
   ├─→ Limit context to last 10 messages
   └─→ Limit document content to 2000 chars

2. FILE PROCESSING
   ├─→ MD5 hash for duplicate detection
   ├─→ Prevent re-uploading same file
   └─→ Incremental file hashing (4KB chunks)

3. AGENT EXECUTION
   ├─→ Mode-based max_iter limits
   ├─→ Verbose logging only when needed
   └─→ Early termination for simple queries

4. FRONTEND
   ├─→ Minimal dependencies (no frameworks)
   ├─→ CSS loaded once
   └─→ JavaScript event delegation
```

---

## 🚀 Deployment Flow

```
LOCAL DEVELOPMENT
  │
  ├─→ git init
  ├─→ git add .
  ├─→ git commit -m "Initial commit"
  ├─→ git push to GitHub
  │
  ▼
RAILWAY DEPLOYMENT
  │
  ├─→ Connect GitHub repo
  ├─→ Auto-detect Flask app
  ├─→ Read runtime.txt → Python 3.12.8
  ├─→ Read Procfile → gunicorn app:app
  ├─→ Install requirements.txt
  │   ├─ Flask + extensions
  │   ├─ CrewAI 0.86.0
  │   ├─ Document processors
  │   └─ Gunicorn
  │
  ├─→ Set environment variables:
  │   ├─ OPENAI_API_KEY
  │   ├─ SERPAPI_API_KEY
  │   ├─ SECRET_KEY
  │   └─ DATABASE_URL (PostgreSQL)
  │
  ├─→ Run database migrations
  │   └─→ db.create_all() in app context
  │
  ├─→ Start Gunicorn server
  │   └─→ gunicorn app:app
  │
  └─→ Provide public URL
      └─→ https://your-app.railway.app
```

---

**End of Flowchart Documentation**
