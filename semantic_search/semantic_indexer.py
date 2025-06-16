"""
Standalone Semantic Indexer for Vietnamese Historical Texts
Generates embeddings using PhoBERT for Vietnamese text from CSV data
"""

import numpy as np
import pandas as pd
import json
import pickle
import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from config import (
    SEMANTIC_MODEL_NAME, SEMANTIC_MODEL_CACHE_DIR, EMBEDDING_DIMENSION,
    SEMANTIC_BATCH_SIZE, MAX_SEQUENCE_LENGTH, FAISS_INDEX_TYPE,
    ENABLE_GPU_ACCELERATION, SEMANTIC_INDEX_DIR, SEMANTIC_EMBEDDINGS_FILE,
    SEMANTIC_FAISS_FILE, SEMANTIC_METADATA_FILE, CSV_DATA_PATH
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticIndexer:
    """Generates and manages semantic embeddings for documents from CSV data"""

    def __init__(self, model_name: str = SEMANTIC_MODEL_NAME,
                 cache_dir: str = SEMANTIC_MODEL_CACHE_DIR):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() and ENABLE_GPU_ACCELERATION else 'cpu')

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.embeddings = {}  # doc_id -> embedding
        self.document_texts = {}  # doc_id -> text
        self.documents = {}  # doc_id -> full document data
        self.faiss_index = None
        self.doc_id_to_index = {}  # doc_id -> faiss index
        self.index_to_doc_id = {}  # faiss index -> doc_id

        logger.info(f"Standalone semantic indexer initialized with device: {self.device}")

    def load_documents_from_csv(self, csv_path: str = None, max_docs: int = None) -> Dict[str, Dict]:
        """Load documents from CSV file"""
        csv_path = csv_path or CSV_DATA_PATH
        logger.info(f"Loading documents from CSV: {csv_path}")

        # Load CSV data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} rows")

        # Limit documents if specified
        if max_docs and max_docs < len(df):
            df = df.head(max_docs)
            logger.info(f"Limited to {max_docs} documents")

        # Process documents
        documents = {}
        for idx, row in df.iterrows():
            doc_id = str(idx)
            title = str(row.get('title', '')) if pd.notna(row.get('title')) else ""
            content = str(row.get('content', '')) if pd.notna(row.get('content')) else ""
            url = str(row.get('url', '')) if pd.notna(row.get('url')) else ""

            # Skip documents with no meaningful content
            if not title.strip() and not content.strip():
                continue

            documents[doc_id] = {
                'id': doc_id,
                'title': title.strip(),
                'content': content.strip(),
                'url': url.strip(),
                'processed_at': datetime.now().isoformat(),
                'source': 'csv'
            }

        self.documents = documents
        logger.info(f"Successfully processed {len(documents)} documents")

        return documents

    def _load_model(self) -> None:
        """Load PhoBERT model and tokenizer"""
        if self.tokenizer is None or self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            
            # Create cache directory
            os.makedirs(self.cache_dir, exist_ok=True)
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    cache_dir=self.cache_dir
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Model loaded successfully on {self.device}")
                
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                logger.info("Falling back to multilingual BERT...")
                
                # Fallback to multilingual BERT
                self.model_name = "bert-base-multilingual-cased"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                logger.info("Fallback model loaded successfully")
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into embedding"""
        self._load_model()
        
        # Tokenize and truncate
        inputs = self.tokenizer(
            text,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.flatten()
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into embeddings"""
        self._load_model()
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def process_documents(self, documents: Dict[str, Dict]) -> None:
        """Process documents and generate embeddings"""
        logger.info(f"Processing {len(documents)} documents for semantic indexing...")
        
        doc_ids = list(documents.keys())
        texts = []
        
        # Prepare texts for embedding
        for doc_id in doc_ids:
            doc = documents[doc_id]
            title = doc.get('title', '')
            content = doc.get('content', '')
            # Combine title and content with special separator
            full_text = f"{title} [SEP] {content}"
            texts.append(full_text)
            self.document_texts[doc_id] = full_text
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), SEMANTIC_BATCH_SIZE):
            batch_texts = texts[i:i + SEMANTIC_BATCH_SIZE]
            batch_doc_ids = doc_ids[i:i + SEMANTIC_BATCH_SIZE]
            
            logger.info(f"Processing batch {i//SEMANTIC_BATCH_SIZE + 1}/{(len(texts)-1)//SEMANTIC_BATCH_SIZE + 1}")
            
            # Generate embeddings for batch
            batch_embeddings = self._encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            # Store individual embeddings
            for j, doc_id in enumerate(batch_doc_ids):
                self.embeddings[doc_id] = batch_embeddings[j]
        
        # Combine all embeddings
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            self._build_faiss_index(combined_embeddings, doc_ids)
        
        logger.info(f"Semantic indexing completed. Generated {len(self.embeddings)} embeddings.")
    
    def _build_faiss_index(self, embeddings: np.ndarray, doc_ids: List[str]) -> None:
        """Build FAISS index for fast similarity search"""
        logger.info("Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        if FAISS_INDEX_TYPE == "IndexFlatIP":
            self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        else:
            # Default to flat L2 index
            self.faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        
        # Add embeddings to index
        self.faiss_index.add(embeddings.astype(np.float32))
        
        # Create mapping between doc_ids and FAISS indices
        for i, doc_id in enumerate(doc_ids):
            self.doc_id_to_index[doc_id] = i
            self.index_to_doc_id[i] = doc_id
        
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def search_similar(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Search for semantically similar documents"""
        if self.faiss_index is None:
            logger.warning("FAISS index not built. Cannot perform semantic search.")
            return []
        
        # Encode query
        query_embedding = self._encode_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, limit)
        
        # Convert results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.index_to_doc_id:
                doc_id = self.index_to_doc_id[idx]
                results.append((doc_id, float(score)))
        
        return results
    
    def save_indexes(self, output_dir: str = None) -> None:
        """Save semantic indexes to files"""
        output_dir = output_dir or SEMANTIC_INDEX_DIR
        os.makedirs(output_dir, exist_ok=True)

        # Save embeddings and metadata
        embeddings_file = os.path.join(output_dir, SEMANTIC_EMBEDDINGS_FILE)
        with open(embeddings_file, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'document_texts': self.document_texts,
                'documents': self.documents,
                'doc_id_to_index': self.doc_id_to_index,
                'index_to_doc_id': self.index_to_doc_id,
                'model_name': self.model_name,
                'created_at': datetime.now().isoformat(),
                'total_documents': len(self.documents)
            }, f)

        # Save FAISS indexq
        if self.faiss_index is not None:
            faiss_file = os.path.join(output_dir, SEMANTIC_FAISS_FILE)
            faiss.write_index(self.faiss_index, faiss_file)

        # Save metadata as JSON
        metadata_file = os.path.join(output_dir, SEMANTIC_METADATA_FILE)
        metadata = {
            'model_name': self.model_name,
            'embedding_dimension': EMBEDDING_DIMENSION,
            'total_documents': len(self.documents),
            'total_embeddings': len(self.embeddings),
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
            'created_at': datetime.now().isoformat(),
            'csv_source': self.csv_loader.csv_path if hasattr(self.csv_loader, 'csv_path') else 'unknown'
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Semantic indexes saved to {output_dir}")
        logger.info(f"  - Embeddings: {embeddings_file}")
        logger.info(f"  - FAISS index: {faiss_file if self.faiss_index else 'None'}")
        logger.info(f"  - Metadata: {metadata_file}")
    
    def load_indexes(self, input_dir: str = None) -> None:
        """Load semantic indexes from files"""
        input_dir = input_dir or SEMANTIC_INDEX_DIR

        try:
            # Load embeddings and metadata
            embeddings_file = os.path.join(input_dir, SEMANTIC_EMBEDDINGS_FILE)
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.document_texts = data['document_texts']
                self.documents = data.get('documents', {})
                self.doc_id_to_index = data['doc_id_to_index']
                self.index_to_doc_id = data['index_to_doc_id']
                self.model_name = data.get('model_name', SEMANTIC_MODEL_NAME)

            # Load FAISS index
            faiss_file = os.path.join(input_dir, SEMANTIC_FAISS_FILE)
            if os.path.exists(faiss_file):
                self.faiss_index = faiss.read_index(faiss_file)

            # Load metadata if available
            metadata_file = os.path.join(input_dir, SEMANTIC_METADATA_FILE)
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    logger.info(f"Loaded semantic indexes created at: {metadata.get('created_at', 'unknown')}")
                    logger.info(f"Total documents: {metadata.get('total_documents', 'unknown')}")
                    logger.info(f"FAISS index size: {metadata.get('faiss_index_size', 'unknown')}")

            logger.info(f"Semantic indexes loaded from {input_dir}")
            logger.info(f"  - Embeddings: {len(self.embeddings)} documents")
            logger.info(f"  - Documents: {len(self.documents)} documents")
            logger.info(f"  - FAISS index: {'loaded' if self.faiss_index else 'not found'}")

        except FileNotFoundError as e:
            logger.error(f"Semantic index files not found in {input_dir}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading semantic indexes: {e}")
            raise
