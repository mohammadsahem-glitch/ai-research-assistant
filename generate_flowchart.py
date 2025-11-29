"""
AI Research Assistant - Interactive Flowchart Generator
Generates professional flowcharts using Graphviz
"""

from graphviz import Digraph

def create_system_architecture():
    """Create overall system architecture flowchart"""
    dot = Digraph('AI_Assistant_Architecture', comment='AI Research Assistant - System Architecture')
    dot.attr(rankdir='TB', size='12,16')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')

    # Frontend Layer
    with dot.subgraph(name='cluster_0') as c:
        c.attr(style='filled', color='lightgrey', label='Frontend Layer')
        c.node('login', 'login.html\n(Login Page)')
        c.node('register', 'register.html\n(Register Page)')
        c.node('index', 'index.html\n(Chat Interface)')

    # Backend Layer
    with dot.subgraph(name='cluster_1') as c:
        c.attr(style='filled', color='lightgreen', label='Flask Application (app.py)')
        c.node('auth_routes', 'Authentication Routes\n/login\n/register\n/logout')
        c.node('chat_route', 'Chat Route\n/chat (POST)')
        c.node('session_routes', 'Session Routes\n/sessions\n/session/new\n/session/:id')
        c.node('doc_routes', 'Document Routes\n/upload\n/documents')

    # Processing Layer
    with dot.subgraph(name='cluster_2') as c:
        c.attr(style='filled', color='lightyellow', label='AI Processing Engine')
        c.node('crewai', 'CrewAI Agent\nGPT-4o-mini')
        c.node('serpapi', 'SerpAPI\nWeb Search')
        c.node('doc_processor', 'Document Processor\nPDF/Word/Excel/PPT/OCR')
        c.node('formatter', 'Response Formatter\nTables/News/Markdown')

    # Database Layer
    with dot.subgraph(name='cluster_3') as c:
        c.attr(style='filled', color='lightcoral', label='Database Layer (SQLAlchemy + SQLite/PostgreSQL)')
        c.node('user_model', 'User\n(email, password)')
        c.node('session_model', 'ChatSession\n(title, timestamps)')
        c.node('message_model', 'Message\n(role, content)')
        c.node('doc_model', 'UploadedDocument\n(filename, content, hash)')

    # Connections
    dot.edge('login', 'auth_routes', label='POST credentials')
    dot.edge('register', 'auth_routes', label='POST email/password')
    dot.edge('index', 'chat_route', label='POST message')
    dot.edge('index', 'session_routes', label='GET/POST/DELETE')
    dot.edge('index', 'doc_routes', label='POST file')

    dot.edge('chat_route', 'crewai', label='Execute agent')
    dot.edge('crewai', 'serpapi', label='Search if needed')
    dot.edge('chat_route', 'doc_processor', label='Extract text')
    dot.edge('crewai', 'formatter', label='Format response')

    dot.edge('auth_routes', 'user_model', label='Create/Query user')
    dot.edge('session_routes', 'session_model', label='CRUD sessions')
    dot.edge('chat_route', 'message_model', label='Save messages')
    dot.edge('doc_routes', 'doc_model', label='Store documents')

    return dot


def create_chat_flow():
    """Create detailed chat message flow"""
    dot = Digraph('Chat_Flow', comment='Chat Message Processing Flow')
    dot.attr(rankdir='TB', size='10,14')
    dot.attr('node', shape='box', style='filled')

    # Define nodes with colors
    dot.node('start', 'User sends message', fillcolor='lightgreen', shape='ellipse')
    dot.node('validate', 'Validate input\n(not empty?)', fillcolor='lightyellow', shape='diamond')
    dot.node('get_session', 'Get or create\nChatSession', fillcolor='lightblue')
    dot.node('save_user_msg', 'Save user message\nto database', fillcolor='lightblue')
    dot.node('load_context', 'Load conversation\ncontext\n(last 10 messages)', fillcolor='lightblue')
    dot.node('check_rag', 'RAG enabled?', fillcolor='lightyellow', shape='diamond')
    dot.node('load_docs', 'Load user documents\n(max 2000 chars each)', fillcolor='lightblue')
    dot.node('apply_mode', 'Apply agent mode\n(summarize/balanced/\nverbose/deep)', fillcolor='lightblue')
    dot.node('create_agent', 'Create CrewAI Agent\nwith SerpAPI tool', fillcolor='lightcoral')
    dot.node('execute', 'Execute agent.kickoff()', fillcolor='lightcoral')
    dot.node('format', 'Format response\n(tables, news, markdown)', fillcolor='lightgreen')
    dot.node('save_response', 'Save assistant message\nto database', fillcolor='lightblue')
    dot.node('return_json', 'Return JSON response\nto frontend', fillcolor='lightgreen', shape='ellipse')
    dot.node('error', 'Return error 400', fillcolor='red', shape='ellipse')

    # Connections
    dot.edge('start', 'validate')
    dot.edge('validate', 'get_session', label='Valid')
    dot.edge('validate', 'error', label='Invalid')
    dot.edge('get_session', 'save_user_msg')
    dot.edge('save_user_msg', 'load_context')
    dot.edge('load_context', 'check_rag')
    dot.edge('check_rag', 'load_docs', label='Yes')
    dot.edge('check_rag', 'apply_mode', label='No')
    dot.edge('load_docs', 'apply_mode')
    dot.edge('apply_mode', 'create_agent')
    dot.edge('create_agent', 'execute')
    dot.edge('execute', 'format')
    dot.edge('format', 'save_response')
    dot.edge('save_response', 'return_json')

    return dot


def create_authentication_flow():
    """Create user authentication flow"""
    dot = Digraph('Auth_Flow', comment='User Authentication Flow')
    dot.attr(rankdir='TB', size='8,10')
    dot.attr('node', shape='box', style='filled')

    dot.node('visit', 'User visits app', fillcolor='lightgreen', shape='ellipse')
    dot.node('check_auth', 'Is authenticated?', fillcolor='lightyellow', shape='diamond')
    dot.node('home', 'Redirect to /home\n(Chat Interface)', fillcolor='lightgreen', shape='ellipse')
    dot.node('login_page', 'Show /login page', fillcolor='lightblue')
    dot.node('choice', 'User action?', fillcolor='lightyellow', shape='diamond')
    dot.node('register_page', 'Show /register page', fillcolor='lightblue')
    dot.node('register_submit', 'Submit email + password', fillcolor='lightblue')
    dot.node('check_email', 'Email exists?', fillcolor='lightyellow', shape='diamond')
    dot.node('hash_password', 'Hash password\n(Werkzeug)', fillcolor='lightcoral')
    dot.node('create_user', 'Create User record', fillcolor='lightblue')
    dot.node('auto_login', 'Auto-login user\n(Flask-Login)', fillcolor='lightcoral')
    dot.node('login_submit', 'Submit credentials', fillcolor='lightblue')
    dot.node('validate_creds', 'Validate email\n+ password', fillcolor='lightyellow', shape='diamond')
    dot.node('login_user', 'Login user\n(Flask-Login)', fillcolor='lightcoral')
    dot.node('error_exists', 'Error: Email\nalready registered', fillcolor='red', shape='ellipse')
    dot.node('error_invalid', 'Error: Invalid\ncredentials', fillcolor='red', shape='ellipse')

    # Connections
    dot.edge('visit', 'check_auth')
    dot.edge('check_auth', 'home', label='Yes')
    dot.edge('check_auth', 'login_page', label='No')
    dot.edge('login_page', 'choice')
    dot.edge('choice', 'register_page', label='Register')
    dot.edge('choice', 'login_submit', label='Login')

    # Register flow
    dot.edge('register_page', 'register_submit')
    dot.edge('register_submit', 'check_email')
    dot.edge('check_email', 'error_exists', label='Yes')
    dot.edge('check_email', 'hash_password', label='No')
    dot.edge('hash_password', 'create_user')
    dot.edge('create_user', 'auto_login')
    dot.edge('auto_login', 'home')

    # Login flow
    dot.edge('login_submit', 'validate_creds')
    dot.edge('validate_creds', 'login_user', label='Valid')
    dot.edge('validate_creds', 'error_invalid', label='Invalid')
    dot.edge('login_user', 'home')

    return dot


def create_database_schema():
    """Create database relationship diagram"""
    dot = Digraph('Database_Schema', comment='Database Schema & Relationships')
    dot.attr(rankdir='LR', size='10,8')
    dot.attr('node', shape='record', style='filled')

    # Define tables
    dot.node('User', '{User|+ id (PK)\\l+ email (UNIQUE)\\l+ password_hash\\l+ created_at\\l}', fillcolor='lightblue')
    dot.node('ChatSession', '{ChatSession|+ id (PK)\\l+ user_id (FK)\\l+ title\\l+ created_at\\l+ updated_at\\l}', fillcolor='lightgreen')
    dot.node('Message', '{Message|+ id (PK)\\l+ session_id (FK)\\l+ role\\l+ content\\l+ timestamp\\l}', fillcolor='lightyellow')
    dot.node('UploadedDocument', '{UploadedDocument|+ id (PK)\\l+ user_id (FK)\\l+ filename\\l+ file_hash (UNIQUE)\\l+ content\\l+ file_type\\l+ uploaded_at\\l}', fillcolor='lightcoral')

    # Relationships
    dot.edge('User', 'ChatSession', label='1:Many\n(cascade delete)', arrowhead='crow')
    dot.edge('ChatSession', 'Message', label='1:Many\n(cascade delete)', arrowhead='crow')
    dot.edge('User', 'UploadedDocument', label='1:Many\n(cascade delete)', arrowhead='crow')

    return dot


def create_document_upload_flow():
    """Create document upload and processing flow"""
    dot = Digraph('Document_Upload_Flow', comment='Document Upload & Processing Flow')
    dot.attr(rankdir='TB', size='8,12')
    dot.attr('node', shape='box', style='filled')

    dot.node('start', 'User selects file', fillcolor='lightgreen', shape='ellipse')
    dot.node('validate_file', 'Validate file\n(type, size)', fillcolor='lightyellow', shape='diamond')
    dot.node('save_temp', 'Save to\nuploads/ folder', fillcolor='lightblue')
    dot.node('calc_hash', 'Calculate MD5 hash', fillcolor='lightblue')
    dot.node('check_dup', 'Duplicate exists?', fillcolor='lightyellow', shape='diamond')
    dot.node('delete_temp', 'Delete temp file', fillcolor='lightblue')
    dot.node('error_dup', 'Return "Already uploaded"', fillcolor='orange', shape='ellipse')
    dot.node('detect_type', 'Detect file type', fillcolor='lightyellow', shape='diamond')
    dot.node('extract_pdf', 'Extract text\n(PyPDF2)', fillcolor='lightcoral')
    dot.node('extract_word', 'Extract text\n(python-docx)', fillcolor='lightcoral')
    dot.node('extract_ppt', 'Extract text\n(python-pptx)', fillcolor='lightcoral')
    dot.node('extract_excel', 'Extract text\n(openpyxl)', fillcolor='lightcoral')
    dot.node('extract_image', 'OCR text\n(pytesseract)', fillcolor='lightcoral')
    dot.node('extract_text', 'Read text file', fillcolor='lightcoral')
    dot.node('save_db', 'Save to\nUploadedDocument\ntable', fillcolor='lightblue')
    dot.node('success', 'Return success', fillcolor='lightgreen', shape='ellipse')
    dot.node('error_type', 'Error: Invalid\nfile type', fillcolor='red', shape='ellipse')

    # Connections
    dot.edge('start', 'validate_file')
    dot.edge('validate_file', 'save_temp', label='Valid')
    dot.edge('validate_file', 'error_type', label='Invalid')
    dot.edge('save_temp', 'calc_hash')
    dot.edge('calc_hash', 'check_dup')
    dot.edge('check_dup', 'delete_temp', label='Yes')
    dot.edge('delete_temp', 'error_dup')
    dot.edge('check_dup', 'detect_type', label='No')

    # File type branching
    dot.edge('detect_type', 'extract_pdf', label='PDF')
    dot.edge('detect_type', 'extract_word', label='Word')
    dot.edge('detect_type', 'extract_ppt', label='PPT')
    dot.edge('detect_type', 'extract_excel', label='Excel')
    dot.edge('detect_type', 'extract_image', label='Image')
    dot.edge('detect_type', 'extract_text', label='Text')

    # All converge to save
    dot.edge('extract_pdf', 'save_db')
    dot.edge('extract_word', 'save_db')
    dot.edge('extract_ppt', 'save_db')
    dot.edge('extract_excel', 'save_db')
    dot.edge('extract_image', 'save_db')
    dot.edge('extract_text', 'save_db')
    dot.edge('save_db', 'success')

    return dot


def generate_all_flowcharts():
    """Generate all flowcharts and save as PNG and SVG"""

    print("🎨 Generating AI Research Assistant Flowcharts...")
    print("=" * 60)

    flowcharts = [
        ('system_architecture', create_system_architecture(), 'System Architecture'),
        ('chat_flow', create_chat_flow(), 'Chat Message Processing Flow'),
        ('auth_flow', create_authentication_flow(), 'Authentication Flow'),
        ('database_schema', create_database_schema(), 'Database Schema'),
        ('document_upload_flow', create_document_upload_flow(), 'Document Upload Flow')
    ]

    for filename, flowchart, description in flowcharts:
        print(f"\n📊 Creating: {description}")

        # Save as PNG
        flowchart.format = 'png'
        flowchart.render(f'flowcharts/{filename}', cleanup=True)
        print(f"   ✓ Saved: flowcharts/{filename}.png")

        # Save as SVG (scalable)
        flowchart.format = 'svg'
        flowchart.render(f'flowcharts/{filename}', cleanup=True)
        print(f"   ✓ Saved: flowcharts/{filename}.svg")

        # Save source DOT file
        flowchart.save(f'flowcharts/{filename}.gv')
        print(f"   ✓ Saved: flowcharts/{filename}.gv")

    print("\n" + "=" * 60)
    print("✅ All flowcharts generated successfully!")
    print(f"📁 Location: flowcharts/")
    print("\nGenerated files:")
    print("  • system_architecture.png/svg - Overall system diagram")
    print("  • chat_flow.png/svg - Chat processing flow")
    print("  • auth_flow.png/svg - User authentication")
    print("  • database_schema.png/svg - Database relationships")
    print("  • document_upload_flow.png/svg - File upload process")


if __name__ == '__main__':
    import os

    # Create flowcharts directory
    os.makedirs('flowcharts', exist_ok=True)

    # Generate all flowcharts
    generate_all_flowcharts()
