import os
from lightrag import LightRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LightRAG
rag = LightRAG(
    working_dir="./rag_storage"
)

if __name__ == "__main__":
    # Example usage
    print("LightRAG initialized successfully!")
    print("Working directory:", rag.working_dir)

    # You can insert documents
    # rag.insert("Your document text here")

    # And query them
    # result = rag.query("Your question here")
    # print(result)
