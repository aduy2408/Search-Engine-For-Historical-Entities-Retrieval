# Vietnamese Historical Entity Search Engine

A comprehensive search engine for Vietnamese historical texts that combines entity-based search, text search, and semantic search capabilities. The system is designed to help researchers and users explore Vietnamese historical documents through multiple search modalities.

## Features

### Core Search Capabilities
- **Entity Search**: Find documents by historical entities (people, places, organizations)
- **Text Search**: Traditional keyword-based search with TF-IDF ranking
- **Semantic Search**: PhoBERT-powered semantic similarity search for Vietnamese text(NOT DONE)
- **Hybrid Search**: Combines all search types for comprehensive results
- **Type-based Search**: Search by entity types (PER/LOC/ORG)

### Vietnamese Language Support
- **Named Entity Recognition**: Uses `underthesea` for Vietnamese NER
- **Semantic Understanding**: PhoBERT model specifically trained on Vietnamese text
- **Text Preprocessing**: Advanced Vietnamese text normalization and cleaning
- **Stop Words**: Comprehensive Vietnamese stop words handling

### Performance Optimizations
- **Parallel Processing**: Multi-threaded search and indexing
- **Caching**: LRU caches for frequent queries and computations
- **FAISS Integration**: Fast vector similarity search
- **Memory Optimization**: Efficient data structures and memory usage

## Architecture

The system consists of two main components:

### 1. Main Search Engine (`search_engine/`)
- **Entity-based search** with fuzzy matching
- **Text search** with TF-IDF ranking
- **Hybrid search** combining multiple approaches
- **REST API** with web interface

### 2. Semantic Search Module (`semantic_search/`)
- * This is still being worked on
- **Standalone semantic search** using PhoBERT
- **Vector embeddings** with FAISS indexing
- **Independent API** on separate port
- **Direct CSV data loading**

## 📋 Requirements


### Python Dependencies
See `requirements.txt` for complete list.

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd Search_Engine
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build Search Indexes

#### Main Search Engine
```bash
python search_engine/entity_indexer.py
```

#### Semantic Search (Optional, still working on it)
```bash
# Navigate to semantic search directory
cd semantic_search

# Build semantic indexes (downloads PhoBERT model ~400MB)
python build_semantic_indexes.py
```

## Usage

### 1. Start Main Search Engine
```bash
# Start the main search API
python search_engine/api.py
```

### 2. Start Semantic Search (Optional, still working on it)
```bash
# Start semantic search API
cd semantic_search
python semantic_api.py
```

### 3. Web Interface
Open `http://localhost:5000` in your browser for the interactive search interface.

### 4. API Endpoints

#### Main Search Engine
```bash
# Entity search
GET /api/search?q=Tây Ninh&type=entity&limit=10

# Text search
GET /api/search?q=lễ hội&type=text&limit=10

# Hybrid search
GET /api/search?q=văn hóa truyền thống&type=hybrid&limit=10

# Get suggestions
GET /api/suggest?q=Tây&limit=5

```


## Data Format

The system expects CSV data with the following columns:
- `title`: Document title
- `content`: Document content
- `url`: Source URL (optional)


## 🔧 Configuration

### Main Search Engine
Edit `search_engine/config.py`:
```python
# Data paths
DATA_PATH = "Monument_database/processed_vietnamese_texts_combined.csv"
INDEX_DIR = "search_indexes"

# Search settings
DEFAULT_SEARCH_LIMIT = 10
FUZZY_MATCH_THRESHOLD = 0.6

# Performance settings
ENABLE_PARALLEL_PROCESSING = True
MAX_WORKER_THREADS = 4
```

### Semantic Search
Edit `semantic_search/config.py`:
```python
# Model settings
SEMANTIC_MODEL_NAME = "vinai/phobert-base"
EMBEDDING_DIMENSION = 768
SEMANTIC_BATCH_SIZE = 16

# Performance settings
ENABLE_GPU_ACCELERATION = False
```

## 🧪 Testing

### Run Demo
```bash
python search_engine/demo.py
```



