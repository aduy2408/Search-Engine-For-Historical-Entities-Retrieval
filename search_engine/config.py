"""
Configuration settings for Historical Entity Search Engine with Performance Optimizations
"""

# Data paths
DATA_PATH = "Monument_database/processed_vietnamese_texts_combined.csv"
INDEX_DIR = "search_indexes"

# Search settings
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100
FUZZY_MATCH_THRESHOLD = 0.6

# NER processing settings
NER_CHUNK_SIZE = 800  # Size of each text chunk for NER processing
NER_CHUNK_OVERLAP = 100  # Overlap between chunks to capture boundary entities
MAX_CHUNKS_PER_DOCUMENT = 25  # Maximum number of chunks to process per document
ENTITY_TEXT_LIMIT = 800  # Deprecated: kept for backward compatibility

# Performance optimization settings
ENABLE_PARALLEL_PROCESSING = True  # Enable parallel processing for non-NER operations
MAX_WORKER_THREADS = 4  # Maximum number of worker threads
BATCH_SIZE = 200  # Increased batch size for better performance
CACHE_SIZE_SIMILARITY = 5000  # Increased cache size for similarity calculations
CACHE_SIZE_SEARCH = 1000  # Cache size for search results
CACHE_SIZE_SUGGESTIONS = 500  # Cache size for entity suggestions
CACHE_SIZE_STATS = 100  # Cache size for statistics

# Fuzzy matching optimization settings
FUZZY_MATCH_MAX_CANDIDATES = 500  # Maximum candidates to process for fuzzy matching
FUZZY_MATCH_LENGTH_RATIO_MIN = 0.5  # Minimum length ratio for candidate filtering
FUZZY_MATCH_LENGTH_RATIO_MAX = 2.0  # Maximum length ratio for candidate filtering
FUZZY_MATCH_CHAR_OVERLAP_THRESHOLD = 0.3  # Minimum character overlap ratio
FUZZY_MATCH_TOP_RESULTS = 20  # Maximum fuzzy match results to return

# Memory optimization settings
ENABLE_MEMORY_OPTIMIZATION = True  # Enable memory optimization features
USE_SETS_FOR_INDEXES = True  # Use sets instead of lists for better performance
PRECOMPILE_REGEX = True  # Precompile regex patterns for better performance

# API settings
API_HOST = "0.0.0.0"
API_PORT = 5000
DEBUG_MODE = False  # Disable debug mode for better performance
ENABLE_THREADING = True  # Enable threading in Flask
MAX_QUERY_LENGTH = 1000  # Maximum query length to prevent abuse
MAX_SUGGESTION_QUERY_LENGTH = 100  # Maximum suggestion query length

# Entity types mapping
ENTITY_TYPES = {
    'PER': 'Person',
    'PERSON': 'Person',
    'LOC': 'Location', 
    'LOCATION': 'Location',
    'ORG': 'Organization',
    'ORGANIZATION': 'Organization',
    'MISC': 'Miscellaneous'
}

# Indexing optimization settings
FUZZY_DEDUPLICATION_THRESHOLD = 0.95  # Threshold for fuzzy entity deduplication
ENABLE_ENTITY_CACHING = True  # Enable entity caching
PARALLEL_INDEX_UPDATES = True  # Enable parallel index updates
SAVE_JSON_IN_PARALLEL = True  # Save JSON files in parallel

# Threading settings (Note: underthesea is NOT thread-safe)
UNDERTHESEA_THREAD_SAFE = False  # Keep this False - underthesea is NOT thread-safe
PARALLEL_SEARCH_PROCESSING = True  # Enable parallel processing for search operations
PARALLEL_TEXT_PROCESSING = True  # Enable parallel text processing (non-NER)

# Semantic search has been moved to semantic_search/ folder
# This search engine now focuses only on historical entity retrieval

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ENABLE_PERFORMANCE_LOGGING = True  # Log performance metrics

# Index file settings
USE_PICKLE_FOR_MAIN_INDEX = True  # Use pickle for main index file (faster loading)
PICKLE_PROTOCOL = 4  # Pickle protocol version for better performance
COMPRESS_INDEXES = False  # Disable compression for faster loading

# Search optimization settings
USE_HEAPQ_FOR_TOPK = True  # Use heapq for top-k optimization
TOPK_THRESHOLD_MULTIPLIER = 2  # Use heapq when results > limit * multiplier
ENABLE_EARLY_TERMINATION = True  # Enable early termination for searches
PRELOAD_ENTITY_CACHE = True  # Preload entity cache on startup
