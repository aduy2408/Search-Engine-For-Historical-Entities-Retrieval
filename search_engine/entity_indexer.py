"""
Entity Indexer for Historical Search Engine
Processes documents and builds searchable entity indexes
"""

import pandas as pd
import json
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
from underthesea import ner
import logging
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from config import NER_CHUNK_SIZE, NER_CHUNK_OVERLAP, MAX_CHUNKS_PER_DOCUMENT
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityIndexer:
    """Builds and maintains indexes for historical entities with performance optimizations"""
    
    def __init__(self, data_path: str = None, fuzzy_threshold: float = 0.95, enable_memory_optimization: bool = True):
        self.data_path = data_path
        self.entity_index = defaultdict(set)   # entity_text -> {doc_ids} - using sets for automatic deduplication
        self.type_index = defaultdict(list)    # entity_type -> [entities]
        self.document_entities = {}            # doc_id -> [entities]
        self.entity_metadata = {}             # entity_text -> metadata
        self.documents = {}                   # doc_id -> document data
        self.fuzzy_threshold = fuzzy_threshold # Threshold for fuzzy matching in deduplication (increased to 0.95)
        self.enable_memory_optimization = enable_memory_optimization

        # Enhanced indexing features (conditionally enabled for memory optimization)
        if not enable_memory_optimization:
            self.entity_positions = {}            # entity_text -> {doc_id: [positions]}
            self.entity_contexts = {}             # entity_text -> {doc_id: [contexts]}
            self.entity_relationships = defaultdict(dict)  # entity -> {related_entity: strength}
        else:
            # Memory-optimized: only store essential data
            self.entity_positions = None
            self.entity_contexts = None
            self.entity_relationships = None

        self.document_scores = {}             # doc_id -> quality scores (always keep this)
        
        # Performance optimizations
        self._lock = threading.RLock()  # Thread-safe operations
        self._similarity_cache = {}  # Cache for similarity calculations
        self._entity_cache = set()  # Cache for existing entities
        
        # Precompiled regex patterns for better performance
        self._word_pattern = re.compile(r'\b\w+\b', re.IGNORECASE)
        self._whitespace_pattern = re.compile(r'\s+')

    def _create_text_chunks(self, text: str, chunk_size: int = NER_CHUNK_SIZE,
                           overlap: int = NER_CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks for NER processing with optimizations"""
        if not text or len(text.strip()) == 0:
            return []

        text = text.strip()
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If this is not the last chunk, try to break at word boundary
            if end < len(text):
                # Look for the last space within the chunk to avoid breaking words
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap

            # Prevent infinite loop and limit chunks
            if len(chunks) >= MAX_CHUNKS_PER_DOCUMENT:
                logger.warning(f"Reached maximum chunks limit ({MAX_CHUNKS_PER_DOCUMENT}) for document")
                break

        return chunks

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using cached results"""
        # Create cache key
        key = (text1.lower().strip(), text2.lower().strip())
        
        # Check cache first
        if key in self._similarity_cache:
            return self._similarity_cache[key]
        
        # Calculate and cache result
        similarity = SequenceMatcher(None, key[0], key[1]).ratio()
        
        # Limit cache size
        if len(self._similarity_cache) > 10000:
            # Remove oldest 20% of entries
            keys_to_remove = list(self._similarity_cache.keys())[:2000]
            for k in keys_to_remove:
                del self._similarity_cache[k]
        
        self._similarity_cache[key] = similarity
        return similarity

    def _extract_entity_positions_and_contexts(self, text: str, entities: List[Dict], doc_id: str) -> None:
        """Extract positions and contexts for entities in the text (memory optimized)"""
        # Skip if memory optimization is enabled
        if self.enable_memory_optimization:
            return

        text_lower = text.lower()

        for entity in entities:
            entity_text = entity['text'].lower()
            positions = []
            contexts = []

            # Find all occurrences of the entity in the text
            start = 0
            while True:
                pos = text_lower.find(entity_text, start)
                if pos == -1:
                    break

                positions.append(pos)

                # Extract context (50 characters before and after)
                context_start = max(0, pos - 50)
                context_end = min(len(text), pos + len(entity_text) + 50)
                context = text[context_start:context_end]
                contexts.append(context)

                start = pos + 1

            # Store positions and contexts
            if entity_text not in self.entity_positions:
                self.entity_positions[entity_text] = {}
                self.entity_contexts[entity_text] = {}

            self.entity_positions[entity_text][doc_id] = positions
            self.entity_contexts[entity_text][doc_id] = contexts

    def _calculate_document_quality_score(self, title: str, content: str, entities: List[Dict]) -> float:
        """Calculate a quality score for the document based on various factors with optimizations"""
        score = 0.0

        # Length factors (moderate length is better)
        title_words = len(self._word_pattern.findall(title))
        content_words = len(self._word_pattern.findall(content))
        total_words = title_words + content_words



        # Entity density (entities per 100 words)
        if total_words > 0:
            entity_density = (len(entities) / total_words) * 100
            # Optimal density is 2-8 entities per 100 words
            if 2 <= entity_density <= 8:
                score += 1.0
            elif 1 <= entity_density < 2 or 8 < entity_density <= 12:
                score += 0.5

        # Title informativeness (titles with entities are better)
        title_lower = title.lower()
        title_entities = sum(1 for e in entities if e['text'].lower() in title_lower)
        if title_entities > 0:
            score += min(title_entities * 0.3, 1.5)  # Cap at 1.5

        return score

    def _merge_entities(self, entity_lists: List[List[Dict]]) -> List[Dict]:
        """Merge entities from multiple chunks with conservative deduplication and optimizations"""
        if not entity_lists:
            return []

        all_entities = []
        for entities in entity_lists:
            all_entities.extend(entities)

        if not all_entities:
            return []

        # Group entities by type for better deduplication
        entities_by_type = defaultdict(list)
        for entity in all_entities:
            entities_by_type[entity['type']].append(entity)

        merged_entities = []

        for entity_type, entities in entities_by_type.items():
            # Remove exact duplicates first using set
            unique_entities = []
            seen_texts = set()

            for entity in entities:
                entity_text = entity['text'].lower().strip()
                if entity_text not in seen_texts:
                    seen_texts.add(entity_text)
                    unique_entities.append(entity)

            # Conservative deduplication - only merge very similar entities
            final_entities = []
            unique_entities.sort(key=lambda x: len(x['text']), reverse=True)  # Longest first

            for entity in unique_entities:
                entity_text = entity['text'].lower().strip()
                should_add = True

                for existing in final_entities:
                    existing_text = existing['text'].lower().strip()

                    # Only merge if one is clearly a substring AND they're very similar in length
                    if entity_text in existing_text and entity_text != existing_text:
                        # Only merge if the shorter one is at least 80% of the longer one's length
                        length_ratio = len(entity_text) / len(existing_text)
                        if length_ratio >= 0.8:
                            should_add = False
                            break
                    elif existing_text in entity_text and entity_text != existing_text:
                        # Current entity contains existing, replace only if similar length
                        length_ratio = len(existing_text) / len(entity_text)
                        if length_ratio >= 0.8:
                            final_entities.remove(existing)
                            break

                    # Very conservative fuzzy matching - only for very high similarity
                    if len(entity_text) > 3 and len(existing_text) > 3:  # Skip very short entities
                        similarity = self._calculate_similarity(entity_text, existing_text)
                        if similarity >= self.fuzzy_threshold:  # Now 0.95 instead of 0.85
                            # Only merge if they're almost identical
                            if len(entity_text) > len(existing_text):
                                final_entities.remove(existing)
                                break
                            else:
                                should_add = False
                                break

                if should_add:
                    final_entities.append(entity)

            merged_entities.extend(final_entities)

        return merged_entities
        
    def _extract_entities_from_chunk(self, text: str) -> List[Dict]:
        """Extract entities from a single text chunk using underthesea NER with optimizations"""
        if not text or len(text.strip()) == 0:
            return []

        try:
            # NOTE: underthesea is NOT thread-safe, so we don't parallelize this
            ner_results = ner(text)

            entities = []
            current_entity = []
            current_type = None

            for result in ner_results:
                # Handle the 4-tuple format: (word, pos_tag, chunk_tag, ner_tag)
                if isinstance(result, tuple):
                    if len(result) == 4:
                        word, pos_tag, chunk_tag, ner_tag = result
                    elif len(result) == 3:
                        word, pos_tag, ner_tag = result
                    elif len(result) == 2:
                        word, ner_tag = result
                    else:
                        continue
                else:
                    continue

                # Process NER tags (B-*, I-*, O)
                if ner_tag.startswith('B-'):  # Beginning of entity
                    # Save previous entity if exists
                    if current_entity and current_type:
                        entity_text = ' '.join(current_entity).strip()
                        if len(entity_text) > 1:  # Skip single characters
                            entities.append({
                                'text': entity_text,
                                'type': current_type
                            })

                    # Start new entity
                    current_entity = [word]
                    current_type = ner_tag[2:]  # Remove 'B-' prefix

                elif ner_tag.startswith('I-') and current_type == ner_tag[2:]:
                    # Continue current entity
                    current_entity.append(word)

                else:
                    # End current entity (O tag or different entity type)
                    if current_entity and current_type:
                        entity_text = ' '.join(current_entity).strip()
                        if len(entity_text) > 1:
                            entities.append({
                                'text': entity_text,
                                'type': current_type
                            })
                    current_entity = []
                    current_type = None

            # Don't forget the last entity
            if current_entity and current_type:
                entity_text = ' '.join(current_entity).strip()
                if len(entity_text) > 1:
                    entities.append({
                        'text': entity_text,
                        'type': current_type
                    })

            return entities

        except Exception as e:
            logger.error(f"Error in NER for chunk: {e}")
            return []

    def extract_entities_fixed(self, text: str) -> List[Dict]:
        """Extract entities using underthesea NER with chunking for long texts"""
        if not text or len(text.strip()) == 0:
            return []

        # Create chunks from the text
        chunks = self._create_text_chunks(text)

        if not chunks:
            return []

        # Process each chunk sequentially (underthesea is NOT thread-safe)
        chunk_entities = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)})")
            entities = self._extract_entities_from_chunk(chunk)
            chunk_entities.append(entities)

        # Merge entities from all chunks
        merged_entities = self._merge_entities(chunk_entities)

        logger.debug(f"Processed {len(chunks)} chunks, found {len(merged_entities)} unique entities")
        return merged_entities
    
    def _process_single_document(self, doc_data: Tuple[int, pd.Series]) -> Dict:
        """Process a single document and return its data and entities with optimizations"""
        idx, row = doc_data
        doc_id = str(idx)
        title = str(row['title']) if pd.notna(row['title']) else ""
        content = str(row['content']) if pd.notna(row['content']) else ""
        url = str(row['url']) if pd.notna(row['url']) else ""

        # Store document data
        document = {
            'id': doc_id,
            'title': title,
            'content': content,
            'url': url,
            'processed_at': datetime.now().isoformat()
        }

        # Extract entities from title and content separately for better position tracking
        title_entities = self.extract_entities_fixed(title) if title else []
        content_entities = self.extract_entities_fixed(content) if content else []

        # Mark title entities (they're more important)
        for entity in title_entities:
            entity['in_title'] = True

        for entity in content_entities:
            entity['in_title'] = False

        # Combine and deduplicate
        all_entities = title_entities + content_entities
        entities = self._merge_entities([all_entities])

        # Calculate document quality score
        quality_score = self._calculate_document_quality_score(title, content, entities)

        # Extract positions and contexts
        full_text = f"{title} {content}"
        self._extract_entity_positions_and_contexts(full_text, entities, doc_id)

        return {
            'doc_id': doc_id,
            'document': document,
            'entities': entities,
            'title': title,
            'quality_score': quality_score
        }

    def process_documents(self, df: pd.DataFrame, max_docs: Optional[int] = None, batch_size: int = 10) -> None:
        """Process documents and build entity indexes with batch processing optimizations"""
        logger.info(f"Processing documents for entity indexing...")

        if max_docs:
            df = df.head(max_docs)

        # Process documents in batches for better memory management
        total_docs = len(df)
        logger.info(f"Processing {total_docs} documents in batches of {batch_size}")
        
        processed_count = 0
        
        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            batch_df = df.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} (docs {batch_start+1}-{batch_end})")
            
            # Process batch sequentially (due to underthesea thread safety)
            batch_results = []
            for idx, row in batch_df.iterrows():
                try:
                    result = self._process_single_document((idx, row))
                    batch_results.append(result)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing document {idx}: {e}")
            
            # Add batch results to indexes - this can be done in parallel
            with ThreadPoolExecutor(max_workers=min(4, multiprocessing.cpu_count())) as executor:
                futures = [executor.submit(self._add_document_to_indexes, result) for result in batch_results]
                for future in futures:
                    future.result()  # Wait for completion
            
            # Log progress
            if processed_count % 50 == 0:
                logger.info(f"Processed {processed_count}/{total_docs} documents")

        logger.info(f"Processed {processed_count} documents")
        logger.info(f"Found {len(self.entity_index)} unique entities")
        logger.info(f"Entity types: {list(self.type_index.keys())}")

    def _add_document_to_indexes(self, result: Dict) -> None:
        """Add processed document result to indexes with thread safety"""
        doc_id = result['doc_id']
        document = result['document']
        entities = result['entities']
        title = result['title']
        quality_score = result.get('quality_score', 1.0)

        # Thread-safe updates using lock
        with self._lock:
            # Store document data and quality score
            self.documents[doc_id] = document
            self.document_scores[doc_id] = quality_score

            # Store document entities
            self.document_entities[doc_id] = entities

            # Build entity relationships (co-occurrence) - only if memory optimization is disabled
            if not self.enable_memory_optimization and self.entity_relationships is not None:
                entity_texts = [e['text'].lower().strip() for e in entities]
                for i, entity1 in enumerate(entity_texts):
                    for j, entity2 in enumerate(entity_texts):
                        if i != j:
                            if entity2 not in self.entity_relationships[entity1]:
                                self.entity_relationships[entity1][entity2] = 0
                            self.entity_relationships[entity1][entity2] += 1

            # Build indexes
            for entity in entities:
                entity_text = entity['text'].lower().strip()
                entity_type = entity['type']
                in_title = entity.get('in_title', False)

                # Add to entity index (using set for automatic deduplication)
                self.entity_index[entity_text].add(doc_id)
                self._entity_cache.add(entity_text)

                # Add to type index with enhanced information
                self.type_index[entity_type].append({
                    'text': entity_text,
                    'doc_id': doc_id,
                    'doc_title': title,
                    'in_title': in_title,
                    'doc_quality': quality_score
                })

                # Update entity metadata with enhanced information
                if entity_text not in self.entity_metadata:
                    self.entity_metadata[entity_text] = {
                        'type': entity_type,
                        'frequency': 0,
                        'documents': set(),
                        'title_appearances': 0,
                        'total_quality': 0.0,
                        'avg_doc_quality': 0.0
                    }

                # Update metadata
                meta = self.entity_metadata[entity_text]
                meta['frequency'] += 1
                meta['documents'].add(doc_id)
                meta['total_quality'] += quality_score
                meta['avg_doc_quality'] = meta['total_quality'] / len(meta['documents'])

                if in_title:
                    meta['title_appearances'] += 1
    
    def get_entity_stats(self) -> Dict:
        """Get statistics about extracted entities with optimizations"""
        stats = {
            'total_entities': len(self.entity_index),
            'total_documents': len(self.documents),
            'entity_types': {}
        }
        
        # Use cached entity data
        type_counts = defaultdict(int)
        for entity_type, entities in self.type_index.items():
            type_counts[entity_type] = len(entities)
        
        stats['entity_types'] = dict(type_counts)
        
        # Average entities per document
        if self.documents:
            total_entity_mentions = sum(len(entities) for entities in self.document_entities.values())
            stats['avg_entities_per_doc'] = total_entity_mentions / len(self.documents)
        
        return stats

    def save_indexes(self, output_dir: str = "search_indexes") -> None:
        """Save indexes to files with optimizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert sets to lists for JSON serialization
        entity_index_serializable = {}
        for entity, doc_set in self.entity_index.items():
            entity_index_serializable[entity] = list(doc_set)
        
        # Convert document sets in metadata to lists
        entity_metadata_serializable = {}
        for entity, meta in self.entity_metadata.items():
            meta_copy = meta.copy()
            if 'documents' in meta_copy and isinstance(meta_copy['documents'], set):
                meta_copy['documents'] = list(meta_copy['documents'])
            entity_metadata_serializable[entity] = meta_copy

        # Main indexes file (pickle for efficiency)
        logger.info("Saving main indexes...")
        indexes_data = {
            'entity_index': entity_index_serializable,
            'type_index': dict(self.type_index),
            'document_entities': self.document_entities,
            'entity_metadata': entity_metadata_serializable,
            'documents': self.documents,
            'document_scores': self.document_scores,
            'entity_positions': self.entity_positions,
            'entity_contexts': self.entity_contexts,
            'entity_relationships': dict(self.entity_relationships) if self.entity_relationships else {}
        }
        
        with open(f"{output_dir}/indexes.pkl", 'wb') as f:
            pickle.dump(indexes_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save individual JSON files for debugging/inspection (in parallel)
        def save_json_file(filename, data):
            with open(f"{output_dir}/{filename}", 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(save_json_file, "documents.json", self.documents),
                executor.submit(save_json_file, "entity_metadata.json", entity_metadata_serializable),
                executor.submit(save_json_file, "document_entities.json", self.document_entities),
                executor.submit(save_json_file, "type_index.json", dict(self.type_index)),
                executor.submit(save_json_file, "entity_index.json", entity_index_serializable)
            ]
            
            # Wait for all files to be saved
            for future in futures:
                future.result()

        logger.info(f"Indexes saved to {output_dir}/")
        
        # Log statistics
        stats = self.get_entity_stats()
        logger.info(f"Saved {stats['total_entities']} entities from {stats['total_documents']} documents")
        logger.info(f"Entity types: {stats['entity_types']}")

    def load_indexes(self, input_dir: str = "search_indexes") -> None:
        """Load indexes from files with optimizations"""
        logger.info(f"Loading indexes from {input_dir}/")
        
        # Load main pickle file
        with open(f"{input_dir}/indexes.pkl", 'rb') as f:
            data = pickle.load(f)
        
        # Convert lists back to sets for entity_index
        self.entity_index = defaultdict(set)
        for entity, doc_list in data['entity_index'].items():
            self.entity_index[entity] = set(doc_list)
        
        self.type_index = defaultdict(list, data['type_index'])
        self.document_entities = data['document_entities']
        self.documents = data['documents']
        self.document_scores = data.get('document_scores', {})
        self.entity_positions = data.get('entity_positions', {})
        self.entity_contexts = data.get('entity_contexts', {})
        self.entity_relationships = defaultdict(dict, data.get('entity_relationships', {}))
        
        # Convert document lists back to sets in metadata
        self.entity_metadata = {}
        for entity, meta in data['entity_metadata'].items():
            meta_copy = meta.copy()
            if 'documents' in meta_copy and isinstance(meta_copy['documents'], list):
                meta_copy['documents'] = set(meta_copy['documents'])
            self.entity_metadata[entity] = meta_copy
        
        # Rebuild caches
        self._entity_cache = set(self.entity_index.keys())
        
        logger.info(f"Loaded {len(self.entity_index)} entities from {len(self.documents)} documents")


def main():
    """Main function to run entity indexing with optimizations"""
    logger.info("Starting entity indexing process...")
    
    # Initialize indexer with memory optimization
    indexer = EntityIndexer(enable_memory_optimization=True)
    
    # Load data
    data_path = indexer.data_path or "Monument_database/processed_vietnamese_texts_enhanced_full.csv"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(df)} documents")
    
    # Process documents with batching
    indexer.process_documents(df, batch_size=20)  # Increased batch size for better performance
    
    # Save indexes
    indexer.save_indexes()
    
    # Print final statistics
    stats = indexer.get_entity_stats()
    logger.info("Final Statistics:")
    logger.info(f"  Total entities: {stats['total_entities']}")
    logger.info(f"  Total documents: {stats['total_documents']}")
    logger.info(f"  Average entities per document: {stats.get('avg_entities_per_doc', 0):.2f}")
    logger.info(f"  Entity types: {stats['entity_types']}")
    
    logger.info("Entity indexing completed successfully!")


if __name__ == "__main__":
    main()
