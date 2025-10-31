import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys in .env file")

if not N8N_WEBHOOK_URL:
    print("Warning: N8N_WEBHOOK_URL not set — chat logs won't be sent to Google Sheets.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = "tattva-shanti-rag"
pc = PineconeClient(api_key=PINECONE_API_KEY)

while not pc.describe_index(INDEX_NAME).status["ready"]:
    time.sleep(1)

pinecone_index = pc.Index(INDEX_NAME)