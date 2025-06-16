"""
Configuration settings for Standalone Semantic Search functionality
"""

import os

# Data source settings - CSV files for historical Vietnamese texts
CSV_DATA_PATH = "/home/duyle/Documents/history_seg301m/Monument_database/processed_vietnamese_texts_enhanced_full.csv"
BACKUP_CSV_PATH = "/home/duyle/Documents/history_seg301m/Monument_database/processed_vietnamese_texts_combined.csv"

# Index storage settings
SEMANTIC_INDEX_DIR = "semantic_indexes"  # Separate from main search indexes
SEMANTIC_EMBEDDINGS_FILE = "semantic_embeddings.pkl"
SEMANTIC_FAISS_FILE = "semantic_faiss.index"
SEMANTIC_METADATA_FILE = "semantic_metadata.json"

# Semantic search settings
SEMANTIC_MODEL_NAME = "vinai/phobert-base"  # PhoBERT model for Vietnamese
SEMANTIC_MODEL_CACHE_DIR = "models/phobert"
EMBEDDING_DIMENSION = 768  # PhoBERT embedding dimension
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
SEMANTIC_BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 256

# CSV processing settings
CSV_TITLE_COLUMN = "title"
CSV_CONTENT_COLUMN = "content"
CSV_URL_COLUMN = "url"
CSV_ENCODING = "utf-8"
MAX_DOCUMENTS_TO_PROCESS = None  # None means process all documents

# Vector storage settings
VECTOR_INDEX_TYPE = "faiss"  # Options: faiss, annoy, simple
FAISS_INDEX_TYPE = "IndexFlatIP"  # Inner product for cosine similarity
ENABLE_GPU_ACCELERATION = False  # Set to True if GPU available

# Search settings
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100
MIN_SIMILARITY_SCORE = 0.5

# API settings
API_HOST = "0.0.0.0"
API_PORT = 5001  # Different port from main search engine
DEBUG_MODE = True

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
