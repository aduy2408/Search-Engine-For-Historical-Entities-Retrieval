# Vietnamese Historical Entity Search Engine

A comprehensive search engine for Vietnamese historical texts that combines entity-based search, text search, and semantic search capabilities. The system is designed to help researchers and users explore Vietnamese historical documents through multiple search modalities.

## 🌟 Features

### Core Search Capabilities
- **Entity Search**: Find documents by historical entities (people, places, organizations)
- **Text Search**: Traditional keyword-based search with TF-IDF ranking
- **Semantic Search**: PhoBERT-powered semantic similarity search for Vietnamese text
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

## 🏗️ Architecture

The system consists of two main components:

### 1. Main Search Engine (`search_engine/`)
- **Entity-based search** with fuzzy matching
- **Text search** with TF-IDF ranking
- **Hybrid search** combining multiple approaches
- **REST API** with web interface

### 2. Semantic Search Module (`semantic_search/`)
- **Standalone semantic search** using PhoBERT
- **Vector embeddings** with FAISS indexing
- **Independent API** on separate port
- **Direct CSV data loading**

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB+ recommended for semantic search)
- 2GB+ disk space for models and indexes

### Python Dependencies
See `requirements.txt` for complete list. Key dependencies:
- **Web Framework**: Flask, Flask-CORS
- **Data Processing**: pandas, numpy
- **Vietnamese NLP**: underthesea, transformers
- **Machine Learning**: torch, faiss-cpu
- **Text Processing**: regex, unidecode

## 🚀 Installation

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
# Build entity indexes (requires CSV data)
python search_engine/entity_indexer.py
```

#### Semantic Search (Optional)
```bash
# Navigate to semantic search directory
cd semantic_search

# Build semantic indexes (downloads PhoBERT model ~400MB)
python build_semantic_indexes.py
```

## 🎯 Usage

### 1. Start Main Search Engine
```bash
# Start the main search API
python search_engine/api.py
```
Access at: `http://localhost:5000`

### 2. Start Semantic Search (Optional)
```bash
# Start semantic search API
cd semantic_search
python semantic_api.py
```
Access at: `http://localhost:5001`

### 3. Web Interface
Open `http://localhost:5000` in your browser for the interactive search interface.

### 4. API Endpoints

#### Main Search Engine (Port 5000)
```bash
# Entity search
GET /api/search?q=Tây Ninh&type=entity&limit=10

# Text search
GET /api/search?q=lễ hội&type=text&limit=10

# Hybrid search
GET /api/search?q=văn hóa truyền thống&type=hybrid&limit=10

# Get suggestions
GET /api/suggest?q=Tây&limit=5

# Get statistics
GET /api/stats
```

#### Semantic Search (Port 5001)
```bash
# Semantic search
GET /api/semantic/search?q=văn hóa truyền thống&limit=10

# API status
GET /api/semantic/status
```

### 5. Python API
```python
from search_engine.search_core import HistoricalSearchEngine

# Initialize search engine
engine = HistoricalSearchEngine()
engine.load_indexes()

# Perform searches
entity_results = engine.search_by_entity("Tây Ninh", fuzzy=True, limit=10)
text_results = engine.search_text("lễ hội", limit=10)
hybrid_results = engine.hybrid_search("văn hóa truyền thống", limit=10)
```

## 📊 Data Format

The system expects CSV data with the following columns:
- `title`: Document title
- `content`: Document content
- `url`: Source URL (optional)

Example data location: `Monument_database/processed_vietnamese_texts_combined.csv`

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
# Interactive demo
python search_engine/demo.py
```

### Performance Testing
```bash
# Quick performance test
python search_engine/test_api_performance.py
```

### Semantic Search Testing
```bash
cd semantic_search
python test_system.py
```

## 📁 Project Structure

```
Search_Engine/
├── search_engine/           # Main search engine
│   ├── api.py              # Flask API server
│   ├── search_core.py      # Core search logic
│   ├── entity_indexer.py   # Entity indexing
│   ├── config.py           # Configuration
│   └── demo.py             # Demo script
├── semantic_search/         # Semantic search module
│   ├── semantic_api.py     # Semantic search API
│   ├── semantic_indexer.py # PhoBERT indexing
│   ├── build_semantic_indexes.py
│   └── config.py           # Semantic config
├── Monument_database/       # Data and preprocessing
│   ├── processed_vietnamese_texts_combined.csv
│   └── preprocess_vietnamese.py
├── search_indexes/          # Generated indexes
└── requirements.txt         # Python dependencies
```

## 🔍 Search Examples

### Vietnamese Queries
- **Entity Search**: "Tây Ninh", "Hồ Chí Minh", "Cao Đài"
- **Text Search**: "lễ hội", "văn hóa", "truyền thống"
- **Semantic Search**: "văn hóa truyền thống", "lễ hội dân tộc"
- **Type Search**: "PER" (people), "LOC" (locations), "ORG" (organizations)

## 🚨 Troubleshooting

### Common Issues

1. **Missing Indexes**
   ```bash
   # Build indexes first
   python search_engine/entity_indexer.py
   ```

2. **PhoBERT Model Download Fails**
   ```bash
   # Manual download
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('vinai/phobert-base')"
   ```

3. **Memory Issues**
   ```python
   # Reduce batch size in config
   SEMANTIC_BATCH_SIZE = 8
   ```

4. **FAISS Installation Issues**
   ```bash
   # Try conda instead of pip
   conda install faiss-cpu -c conda-forge
   ```

## 📈 Performance

### Benchmarks
- **Entity Search**: <50ms for typical queries
- **Text Search**: <100ms for typical queries
- **Semantic Search**: <200ms after model loading
- **Index Loading**: 2-5 seconds for main indexes
- **PhoBERT Loading**: 5-10 seconds (first time)

### Optimization Tips
1. **Use SSD storage** for faster index loading
2. **Increase RAM** for better caching
3. **Enable GPU** for semantic search if available
4. **Tune batch sizes** based on available memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **underthesea**: Vietnamese NLP toolkit
- **PhoBERT**: Vietnamese BERT model by VinAI
- **FAISS**: Facebook AI Similarity Search
- **Flask**: Web framework for APIs

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration files
3. Check logs for error messages
4. Open an issue on the repository

---

**Note**: This search engine is optimized for Vietnamese historical texts and uses specialized Vietnamese NLP models for best results.
