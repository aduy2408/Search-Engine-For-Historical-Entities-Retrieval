"""
CSV Data Loader for Standalone Semantic Search
Loads and processes Vietnamese historical text data from CSV files
"""

import pandas as pd
import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from config import (
    CSV_DATA_PATH, BACKUP_CSV_PATH, CSV_TITLE_COLUMN, CSV_CONTENT_COLUMN,
    CSV_URL_COLUMN, CSV_ENCODING, MAX_DOCUMENTS_TO_PROCESS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataLoader:
    """Loads and processes CSV data for semantic search indexing"""
    
    def __init__(self, csv_path: str = None, encoding: str = CSV_ENCODING):
        self.csv_path = csv_path or CSV_DATA_PATH
        self.encoding = encoding
        self.documents = {}  # doc_id -> document data
        self.metadata = {}   # Additional metadata about the dataset
        
    def load_csv_data(self, max_docs: int = None) -> Dict[str, Dict]:
        """Load documents from CSV file"""
        logger.info(f"Loading CSV data from: {self.csv_path}")
        
        try:
            # Try primary CSV path first
            if not os.path.exists(self.csv_path):
                logger.warning(f"Primary CSV file not found: {self.csv_path}")
                logger.info(f"Trying backup CSV file: {BACKUP_CSV_PATH}")
                self.csv_path = BACKUP_CSV_PATH
                
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"No CSV file found at {self.csv_path} or {BACKUP_CSV_PATH}")
            
            # Load CSV data
            df = pd.read_csv(self.csv_path, encoding=self.encoding)
            logger.info(f"Loaded CSV with {len(df)} rows")
            
            # Limit documents if specified
            max_docs = max_docs or MAX_DOCUMENTS_TO_PROCESS
            if max_docs and max_docs < len(df):
                df = df.head(max_docs)
                logger.info(f"Limited to {max_docs} documents")
            
            # Process documents
            documents = self._process_dataframe(df)
            
            # Store metadata
            self.metadata = {
                'csv_path': self.csv_path,
                'total_documents': len(documents),
                'processed_at': datetime.now().isoformat(),
                'columns': list(df.columns),
                'encoding': self.encoding
            }
            
            logger.info(f"Successfully processed {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise
    
    def _process_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Process DataFrame into document format"""
        documents = {}
        
        for idx, row in df.iterrows():
            doc_id = str(idx)
            
            # Extract required fields
            title = str(row.get(CSV_TITLE_COLUMN, '')) if pd.notna(row.get(CSV_TITLE_COLUMN)) else ""
            content = str(row.get(CSV_CONTENT_COLUMN, '')) if pd.notna(row.get(CSV_CONTENT_COLUMN)) else ""
            url = str(row.get(CSV_URL_COLUMN, '')) if pd.notna(row.get(CSV_URL_COLUMN)) else ""
            
            # Skip documents with no meaningful content
            if not title.strip() and not content.strip():
                logger.debug(f"Skipping document {doc_id} - no title or content")
                continue
            
            # Create document
            document = {
                'id': doc_id,
                'title': title.strip(),
                'content': content.strip(),
                'url': url.strip(),
                'processed_at': datetime.now().isoformat(),
                'source': 'csv'
            }
            
            # Add any additional columns as metadata
            additional_fields = {}
            for col in df.columns:
                if col not in [CSV_TITLE_COLUMN, CSV_CONTENT_COLUMN, CSV_URL_COLUMN]:
                    if pd.notna(row.get(col)):
                        additional_fields[col] = str(row[col])
            
            if additional_fields:
                document['additional_fields'] = additional_fields
            
            documents[doc_id] = document
        
        return documents
    
    def get_document_stats(self, documents: Dict[str, Dict]) -> Dict:
        """Get statistics about the loaded documents"""
        if not documents:
            return {}
        
        total_docs = len(documents)
        docs_with_title = sum(1 for doc in documents.values() if doc['title'])
        docs_with_content = sum(1 for doc in documents.values() if doc['content'])
        docs_with_url = sum(1 for doc in documents.values() if doc['url'])
        
        # Calculate text lengths
        title_lengths = [len(doc['title']) for doc in documents.values() if doc['title']]
        content_lengths = [len(doc['content']) for doc in documents.values() if doc['content']]
        
        stats = {
            'total_documents': total_docs,
            'documents_with_title': docs_with_title,
            'documents_with_content': docs_with_content,
            'documents_with_url': docs_with_url,
            'avg_title_length': sum(title_lengths) / len(title_lengths) if title_lengths else 0,
            'avg_content_length': sum(content_lengths) / len(content_lengths) if content_lengths else 0,
            'max_title_length': max(title_lengths) if title_lengths else 0,
            'max_content_length': max(content_lengths) if content_lengths else 0,
        }
        
        return stats
    
    def validate_csv_structure(self, csv_path: str = None) -> Tuple[bool, List[str]]:
        """Validate that CSV has required columns"""
        csv_path = csv_path or self.csv_path
        errors = []
        
        try:
            if not os.path.exists(csv_path):
                errors.append(f"CSV file not found: {csv_path}")
                return False, errors
            
            # Read just the header
            df_header = pd.read_csv(csv_path, nrows=0, encoding=self.encoding)
            columns = list(df_header.columns)
            
            # Check required columns
            required_columns = [CSV_TITLE_COLUMN, CSV_CONTENT_COLUMN]
            for col in required_columns:
                if col not in columns:
                    errors.append(f"Required column '{col}' not found in CSV")
            
            # Check optional columns
            if CSV_URL_COLUMN not in columns:
                logger.warning(f"Optional column '{CSV_URL_COLUMN}' not found in CSV")
            
            if errors:
                return False, errors
            
            logger.info(f"CSV structure validation passed for: {csv_path}")
            logger.info(f"Available columns: {columns}")
            return True, []
            
        except Exception as e:
            errors.append(f"Error validating CSV structure: {e}")
            return False, errors

def main():
    """Test the CSV loader"""
    loader = CSVDataLoader()
    
    # Validate CSV structure
    is_valid, errors = loader.validate_csv_structure()
    if not is_valid:
        print("CSV validation failed:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # Load documents
    documents = loader.load_csv_data(max_docs=10)  # Load first 10 for testing
    
    # Print statistics
    stats = loader.get_document_stats(documents)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Print sample documents
    print(f"\nSample Documents (first 3):")
    for i, (doc_id, doc) in enumerate(list(documents.items())[:3]):
        print(f"\nDocument {doc_id}:")
        print(f"  Title: {doc['title'][:100]}...")
        print(f"  Content: {doc['content'][:200]}...")
        print(f"  URL: {doc['url']}")

if __name__ == "__main__":
    main()
