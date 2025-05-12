# config.py

import os

# Langchain config
LANGSMITH_TRACING = "true"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = "LANGCHAIN_API_KEY"


# Gemini API Key
GEMINI_API_KEY = "GEMINI_API_KEY"

# Paths
PDF_FILE = "data/pfizer-report.pdf"
CHROMA_DB_FOLDER = "embeddings/chroma_db/"
DATA_SAVE_PATH = "outputs/"

# Model names
CHAT_MODEL_NAME = "models/gemini-1.5-flash"
SUMMARY_MODEL_NAME = "models/gemini-1.5-flash"
TEXT_EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl"
TABLE_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


# Chunking
CHUNK_SIZE = 1000  # number of characters per chunk
CHUNK_OVERLAP = 100  # overlap between chunks

# Other settings
EMBEDDING_DEVICE = "gpu"  # or "cuda" (GPU)