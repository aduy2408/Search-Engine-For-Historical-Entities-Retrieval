

import pickle
from collections import defaultdict
from typing import Dict, List, Set, Optional
import re
from difflib import SequenceMatcher
import logging
from functools import lru_cache
import threading
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalSearchEngine:
    """Core search engine for historical entities with performance optimizations"""

    def __init__(self, index_dir: str = "search_indexes"):
        # Handle relative path from search_engine directory to root directory
        import os
        if not os.path.isabs(index_dir) and not os.path.exists(index_dir):
            # Try looking in parent directory (repository root)
            parent_index_dir = os.path.join("..", index_dir)
            if os.path.exists(parent_index_dir):
                self.index_dir = parent_index_dir
            else:
                self.index_dir = index_dir
        else:
            self.index_dir = index_dir
        self.entity_index = defaultdict(set)  # Changed to set for faster lookups
        self.type_index = defaultdict(list)
        self.document_entities = {}
        self.entity_metadata = {}
        self.documents = {}
        self.loaded = False

        # Enhanced indexing features
        self.document_scores = {}
        self.entity_positions = {}
        self.entity_contexts = {}
        self.entity_relationships = {}

        # Performance optimizations
        self._entity_text_cache = set()  # Fast entity existence check
        self._similarity_cache = {}  # Cache for similarity calculations
        self._cache_lock = threading.RLock()  # Thread-safe caching

        # Precompiled regex for better performance
        self._word_pattern = re.compile(r'\b\w+\b', re.IGNORECASE)

        # Advanced ranking features
        self._document_frequencies = {}  # word -> number of documents containing it
        self._total_documents = 0
        self._idf_cache = {}  # word -> IDF score

        # PERFORMANCE OPTIMIZATION: Reuse thread pool to avoid creation overhead
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        self._thread_pool = ThreadPoolExecutor(max_workers=min(4, multiprocessing.cpu_count()))

        # PERFORMANCE OPTIMIZATION: Pre-sorted entity list for faster fuzzy matching
        self._sorted_entities = []
        self._entity_length_index = {}  # length -> [entities] for faster fuzzy matching

        # PERFORMANCE OPTIMIZATION: Simple query result cache
        self._text_search_cache = {}
        self._cache_max_size = 100

    def load_indexes(self) -> None:
        """Load pre-built indexes with optimizations"""
        try:
            with open(f"{self.index_dir}/indexes.pkl", 'rb') as f:
                data = pickle.load(f)
                
                # Convert entity_index to use sets for faster lookups
                entity_index_data = data.get('entity_index', {})
                if isinstance(entity_index_data, dict):
                    for entity, doc_list in entity_index_data.items():
                        if isinstance(doc_list, list):
                            self.entity_index[entity] = set(doc_list)
                        else:
                            self.entity_index[entity] = doc_list
                else:
                    self.entity_index = defaultdict(set, entity_index_data)
                
                self.type_index = defaultdict(list, data['type_index'])
                self.document_entities = data['document_entities']
                self.entity_metadata = data['entity_metadata']
                self.documents = data['documents']

                # Load enhanced features (with backward compatibility)
                self.document_scores = data.get('document_scores', {})
                self.entity_positions = data.get('entity_positions', {})
                self.entity_contexts = data.get('entity_contexts', {})
                self.entity_relationships = data.get('entity_relationships', {})

            # Build performance caches
            self._entity_text_cache = set(self.entity_index.keys())

            # PERFORMANCE OPTIMIZATION: Build sorted entity lists for faster fuzzy matching
            self._build_entity_indexes()

            # Build advanced ranking features
            self._build_document_frequencies()

            self.loaded = True
            logger.info(f"Search indexes loaded successfully with optimizations")
        except FileNotFoundError:
            logger.error(f"Index files not found in {self.index_dir}")
            raise
    
    def _ensure_loaded(self) -> None:
        """Ensure indexes are loaded"""
        if not self.loaded:
            self.load_indexes()

    def _build_entity_indexes(self) -> None:
        """Build optimized entity indexes for faster fuzzy matching"""
        logger.info("Building optimized entity indexes for faster fuzzy matching...")

        # Sort entities by length for more efficient fuzzy matching
        self._sorted_entities = sorted(self._entity_text_cache, key=len)

        # Build length-based index for faster filtering
        self._entity_length_index = {}
        for entity in self._sorted_entities:
            length = len(entity)
            if length not in self._entity_length_index:
                self._entity_length_index[length] = []
            self._entity_length_index[length].append(entity)

        logger.info(f"Built entity indexes for {len(self._sorted_entities)} entities")

    def _build_document_frequencies(self) -> None:
        """Build document frequency statistics for TF-IDF ranking"""
        logger.info("Building document frequency statistics for advanced ranking...")
        self._document_frequencies = defaultdict(int)
        self._total_documents = len(self.documents)

        for doc_id, doc in self.documents.items():
            # Extract words from title and content
            title_words = set(self._word_pattern.findall(doc['title'].lower()))
            content_words = set(self._word_pattern.findall(doc['content'].lower()))
            all_words = title_words.union(content_words)

            # Count document frequency for each word
            for word in all_words:
                self._document_frequencies[word] += 1

        logger.info(f"Built document frequencies for {len(self._document_frequencies)} unique words")

    def _calculate_idf(self, word: str) -> float:
        """Calculate Inverse Document Frequency for a word"""
        if word in self._idf_cache:
            return self._idf_cache[word]

        doc_freq = self._document_frequencies.get(word, 0)
        if doc_freq == 0:
            idf = 0.0
        else:
            idf = math.log(self._total_documents / doc_freq)

        self._idf_cache[word] = idf
        return idf

    def _calculate_tf_idf_score(self, query_words: List[str], doc_title: str, doc_content: str) -> float:
        """Calculate TF-IDF score for a document given query words"""
        title_words = self._word_pattern.findall(doc_title.lower())
        content_words = self._word_pattern.findall(doc_content.lower())

        # Count term frequencies
        title_tf = defaultdict(int)
        content_tf = defaultdict(int)

        for word in title_words:
            title_tf[word] += 1
        for word in content_words:
            content_tf[word] += 1

        total_words = len(title_words) + len(content_words)
        if total_words == 0:
            return 0.0

        tf_idf_score = 0.0
        for query_word in query_words:
            query_word_lower = query_word.lower()

            # Calculate TF (with title boost)
            title_freq = title_tf.get(query_word_lower, 0)
            content_freq = content_tf.get(query_word_lower, 0)

            # Title words get 3x weight, content words get 1x weight
            tf = (title_freq * 3 + content_freq) / total_words

            # Calculate IDF
            idf = self._calculate_idf(query_word_lower)

            # Add to total TF-IDF score
            tf_idf_score += tf * idf

        return tf_idf_score

    def _calculate_query_proximity_score(self, query: str, doc_title: str, doc_content: str) -> float:
        """Calculate proximity score based on how close query terms appear together"""
        query_words = self._word_pattern.findall(query.lower())
        if len(query_words) < 2:
            return 1.0  # No proximity bonus for single words

        # Check title proximity (higher weight)
        title_text = doc_title.lower()
        title_proximity = self._calculate_text_proximity(query_words, title_text)

        # Check content proximity
        content_text = doc_content.lower()
        content_proximity = self._calculate_text_proximity(query_words, content_text)

        # Combine with title getting higher weight
        return (title_proximity * 2 + content_proximity) / 3

    def _calculate_text_proximity(self, query_words: List[str], text: str) -> float:
        """Calculate how close query words appear in text"""
        if len(query_words) < 2:
            return 1.0

        # Find positions of all query words
        word_positions = defaultdict(list)
        words_in_text = self._word_pattern.findall(text)

        for i, word in enumerate(words_in_text):
            if word in query_words:
                word_positions[word].append(i)

        # Check if all query words are present
        found_words = [word for word in query_words if word in word_positions]
        if len(found_words) < 2:
            return 0.5  # Partial match

        # Calculate minimum distance between any pair of query words
        min_distance = float('inf')
        for i, word1 in enumerate(found_words):
            for j, word2 in enumerate(found_words):
                if i != j:
                    for pos1 in word_positions[word1]:
                        for pos2 in word_positions[word2]:
                            distance = abs(pos1 - pos2)
                            min_distance = min(min_distance, distance)

        if min_distance == float('inf'):
            return 0.5

        # Convert distance to proximity score (closer = higher score)
        # Max bonus for words within 5 positions, decreasing bonus up to 50 positions
        if min_distance <= 5:
            return 2.0  # Strong proximity bonus
        elif min_distance <= 20:
            return 1.5  # Medium proximity bonus
        elif min_distance <= 50:
            return 1.2  # Small proximity bonus
        else:
            return 1.0  # No proximity bonus

    def _calculate_entity_context_relevance(self, query_entity: str, matched_entity: str,
                                          doc_title: str, doc_content: str) -> float:
        """Calculate how relevant the entity is in the context of the document"""
        # Check if query terms appear near the entity in the document
        query_words = self._word_pattern.findall(query_entity.lower())
        entity_words = self._word_pattern.findall(matched_entity.lower())

        # Combine title and content for context analysis
        full_text = f"{doc_title} {doc_content}".lower()
        text_words = self._word_pattern.findall(full_text)

        # Find positions of entity words
        entity_positions = []
        for i, word in enumerate(text_words):
            if word in entity_words:
                entity_positions.append(i)

        if not entity_positions:
            return 1.0  # Default score if entity not found

        # Find positions of query words
        query_positions = []
        for i, word in enumerate(text_words):
            if word in query_words and word not in entity_words:  # Exclude entity words from query
                query_positions.append(i)

        if not query_positions:
            return 1.0  # No additional context

        # Calculate minimum distance between entity and query words
        min_distance = float('inf')
        for entity_pos in entity_positions:
            for query_pos in query_positions:
                distance = abs(entity_pos - query_pos)
                min_distance = min(min_distance, distance)

        # Convert to relevance score
        if min_distance <= 10:
            return 1.5  # High relevance - query terms very close to entity
        elif min_distance <= 30:
            return 1.3  # Medium relevance
        elif min_distance <= 100:
            return 1.1  # Low relevance
        else:
            return 1.0  # No proximity relevance

    def _calculate_entity_position_score(self, entity: str, doc_title: str, doc_content: str) -> float:
        """Calculate score based on where the entity appears in the document"""
        entity_lower = entity.lower()

        # Title appearance gets highest score
        if entity_lower in doc_title.lower():
            return 1.5

        # Early appearance in content gets bonus
        content_lower = doc_content.lower()
        entity_pos = content_lower.find(entity_lower)

        if entity_pos == -1:
            return 1.0  # Not found

        content_length = len(content_lower)
        if content_length == 0:
            return 1.0

        # Calculate relative position (0.0 = start, 1.0 = end)
        relative_pos = entity_pos / content_length

        # Earlier positions get higher scores
        if relative_pos <= 0.1:  # First 10%
            return 1.3
        elif relative_pos <= 0.3:  # First 30%
            return 1.2
        elif relative_pos <= 0.5:  # First 50%
            return 1.1
        else:
            return 1.0  # Later in document

    def _calculate_entity_type_relevance(self, query: str, entity_type: str) -> float:
        """Calculate relevance based on entity type and query context"""
        query_lower = query.lower()

        # Define type-specific keywords that suggest relevance
        type_keywords = {
            'PER': ['người', 'ông', 'bà', 'anh', 'chị', 'em', 'tướng', 'vua', 'hoàng', 'chủ tịch', 'thủ tướng'],
            'LOC': ['nơi', 'chỗ', 'địa', 'vùng', 'khu', 'thành phố', 'tỉnh', 'huyện', 'xã', 'đường', 'phố'],
            'ORG': ['tổ chức', 'công ty', 'trường', 'viện', 'bộ', 'ban', 'hội', 'đảng', 'chính phủ'],
            'MISC': ['sự kiện', 'lễ hội', 'ngày', 'năm', 'thời', 'kỷ niệm']
        }

        # Check if query contains keywords relevant to this entity type
        relevant_keywords = type_keywords.get(entity_type, [])
        for keyword in relevant_keywords:
            if keyword in query_lower:
                return 1.3  # Boost for type-relevant queries

        return 1.0  # Default score
    
    @lru_cache(maxsize=5000)  # Increased cache size for better performance
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings with caching"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def __del__(self):
        """Cleanup thread pool when object is destroyed"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)
    
    def _fuzzy_match_entities(self, query: str, threshold: float = 0.7) -> List[str]:
        """Find entities that fuzzy match the query with MAJOR optimizations"""
        self._ensure_loaded()
        query_lower = query.lower().strip()
        query_len = len(query_lower)

        # Quick exact match check first
        if query_lower in self._entity_text_cache:
            return [query_lower]

        # OPTIMIZATION 1: Filter by length to reduce candidates
        # Only check entities within reasonable length range
        min_len = max(1, int(query_len * 0.5))  # At least 50% of query length
        max_len = int(query_len * 2.0)  # At most 200% of query length

        candidates = []
        for length in range(min_len, max_len + 1):
            if length in self._entity_length_index:
                candidates.extend(self._entity_length_index[length])

        # OPTIMIZATION 2: Early filtering with simple string operations
        # Filter candidates that share at least some characters with query
        filtered_candidates = []
        query_chars = set(query_lower)
        for entity in candidates:
            entity_chars = set(entity)
            # Must share at least 30% of characters
            shared_ratio = len(query_chars.intersection(entity_chars)) / len(query_chars.union(entity_chars))
            if shared_ratio >= 0.3:
                filtered_candidates.append(entity)

        # OPTIMIZATION 3: Limit candidates to prevent excessive processing
        if len(filtered_candidates) > 500:  # Limit to top 500 candidates
            # Sort by length similarity first (quick operation)
            filtered_candidates.sort(key=lambda x: abs(len(x) - query_len))
            filtered_candidates = filtered_candidates[:500]

        # OPTIMIZATION 4: Use reusable thread pool for parallel processing
        matches = []

        def calculate_match(entity_text: str) -> Optional[tuple]:
            similarity = self._calculate_similarity(query_lower, entity_text)
            if similarity >= threshold:
                return (entity_text, similarity)
            return None

        # Process in smaller batches to avoid overwhelming the thread pool
        batch_size = 100
        for i in range(0, len(filtered_candidates), batch_size):
            batch = filtered_candidates[i:i + batch_size]
            futures = [self._thread_pool.submit(calculate_match, entity) for entity in batch]

            for future in futures:
                result = future.result()
                if result:
                    matches.append(result)

        # Sort by similarity score and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches[:20]]  # Limit to top 20 matches
    
    def search_by_entity(self, entity_name: str, fuzzy: bool = True, limit: int = 10) -> Dict:
        """Search for documents containing a specific entity with optimizations"""
        self._ensure_loaded()
        
        results = {
            'query': entity_name,
            'search_type': 'entity',
            'fuzzy_enabled': fuzzy,
            'documents': [],
            'total_results': 0,
            'matched_entities': []
        }
        
        entity_name_lower = entity_name.lower().strip()
        matched_entities = []
        
        # Fast exact match first using cache
        if entity_name_lower in self._entity_text_cache:
            matched_entities.append(entity_name_lower)
        
        # Fuzzy match if enabled and no exact match
        if fuzzy and not matched_entities:
            fuzzy_matches = self._fuzzy_match_entities(entity_name, threshold=0.6)
            matched_entities.extend(fuzzy_matches[:5])  # Limit fuzzy matches
        
        # Early return if no matches
        if not matched_entities:
            return results
        
        # Collect documents for matched entities
        doc_scores = defaultdict(float)
        entity_matches = defaultdict(list)
        doc_ranking_details = defaultdict(list)  # Store detailed ranking info
        
        for entity in matched_entities:
            if entity in self.entity_index:
                entity_meta = self.entity_metadata.get(entity, {})
                doc_ids = self.entity_index[entity]  # Now a set for O(1) operations
                
                # Batch process document scoring
                for doc_id in doc_ids:
                    if doc_id not in self.documents:
                        continue

                    doc = self.documents[doc_id]

                    # === BASIC ENTITY SCORING ===
                    entity_freq = entity_meta.get('frequency', 1)
                    similarity = self._calculate_similarity(entity_name_lower, entity)

                    # Base score
                    base_score = entity_freq * similarity

                    # === ADVANCED ENTITY RANKING ===
                    ranking_details = {
                        'base_score': base_score,
                        'entity_frequency': entity_freq,
                        'similarity': similarity,
                        'entity_name': entity,
                        'query_entity': entity_name
                    }

                    # 1. Query-Entity Relevance Score
                    entity_context_score = self._calculate_entity_context_relevance(
                        entity_name, entity, doc['title'], doc['content']
                    )
                    ranking_details['context_relevance'] = entity_context_score

                    # 2. Document quality boost
                    doc_quality = self.document_scores.get(doc_id, 1.0)
                    quality_multiplier = (1 + doc_quality * 0.15)
                    ranking_details['doc_quality'] = doc_quality
                    ranking_details['quality_boost'] = quality_multiplier

                    # 3. Title appearance boost
                    title_appearances = entity_meta.get('title_appearances', 0)
                    title_boost = 2.0 if title_appearances > 0 else 1.0
                    ranking_details['in_title'] = title_appearances > 0
                    ranking_details['title_boost'] = title_boost

                    # 4. Entity position scoring
                    position_score = self._calculate_entity_position_score(entity, doc['title'], doc['content'])
                    ranking_details['position_score'] = position_score

                    # 5. Entity quality boost
                    avg_doc_quality = entity_meta.get('avg_doc_quality', 1.0)
                    entity_quality_boost = (1 + avg_doc_quality * 0.1)
                    ranking_details['entity_quality_boost'] = entity_quality_boost

                    # 6. Exact match bonus
                    if similarity >= 0.95:
                        exact_match_bonus = 1.5
                    elif similarity >= 0.8:
                        exact_match_bonus = 1.2
                    else:
                        exact_match_bonus = 1.0
                    ranking_details['exact_match_bonus'] = exact_match_bonus

                    # 7. Entity type relevance
                    entity_type = entity_meta.get('type', 'UNKNOWN')
                    type_relevance = self._calculate_entity_type_relevance(entity_name, entity_type)
                    ranking_details['type_relevance'] = type_relevance
                    ranking_details['entity_type'] = entity_type

                    # Calculate final score
                    score = (base_score * entity_context_score * quality_multiplier *
                            title_boost * position_score * entity_quality_boost *
                            exact_match_bonus * type_relevance)

                    ranking_details['final_score'] = score

                    doc_scores[doc_id] += score
                    entity_matches[doc_id].append({
                        'entity': entity,
                        'type': entity_meta.get('type', 'UNKNOWN'),
                        'similarity': similarity,
                        'in_title': title_appearances > 0,
                        'quality_score': avg_doc_quality
                    })
                    doc_ranking_details[doc_id].append(ranking_details)
        
        # Sort documents by score - use heapq for top-k optimization
        if len(doc_scores) > limit * 2:
            import heapq
            sorted_docs = heapq.nlargest(limit, doc_scores.items(), key=lambda x: x[1])
        else:
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Build result documents
        for doc_id, score in sorted_docs:
            if doc_id in self.documents:
                doc = self.documents[doc_id].copy()
                doc['score'] = score
                doc['matched_entities'] = entity_matches[doc_id]
                doc['entity_count'] = len(self.document_entities.get(doc_id, []))
                doc['ranking_details'] = doc_ranking_details[doc_id]
                results['documents'].append(doc)
        
        results['total_results'] = len(doc_scores)
        results['matched_entities'] = matched_entities
        
        return results
    
    def search_by_type(self, entity_type: str, limit: int = 10) -> Dict:
        """Search for documents by entity type with optimizations"""
        self._ensure_loaded()
        
        results = {
            'query': entity_type,
            'search_type': 'type',
            'documents': [],
            'total_results': 0,
            'entities_of_type': []
        }
        
        entity_type_upper = entity_type.upper()
        
        if entity_type_upper not in self.type_index:
            return results
        
        # Get all entities of this type
        type_entities = self.type_index[entity_type_upper]
        doc_scores = defaultdict(float)
        doc_entities = defaultdict(list)
        
        # Batch process scoring
        for entity_info in type_entities:
            doc_id = entity_info['doc_id']
            entity_text = entity_info['text']
            
            # Score based on entity frequency
            entity_freq = self.entity_metadata.get(entity_text, {}).get('frequency', 1)
            doc_scores[doc_id] += entity_freq
            doc_entities[doc_id].append(entity_info)
        
        # Use heapq for top-k optimization
        if len(doc_scores) > limit * 2:
            import heapq
            sorted_docs = heapq.nlargest(limit, doc_scores.items(), key=lambda x: x[1])
        else:
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Build result documents
        for doc_id, score in sorted_docs:
            if doc_id in self.documents:
                doc = self.documents[doc_id].copy()
                doc['score'] = score
                doc['entities_of_type'] = doc_entities[doc_id]
                doc['entity_count'] = len(self.document_entities.get(doc_id, []))
                results['documents'].append(doc)
        
        results['total_results'] = len(doc_scores)
        results['entities_of_type'] = list(set([e['text'] for e in type_entities]))
        
        return results
    
    def search_text(self, query: str, limit: int = 10) -> Dict:
        """Search in document titles and content with optimizations"""
        self._ensure_loaded()

        # OPTIMIZATION: Check cache first
        cache_key = f"{query.lower().strip()}:{limit}"
        if cache_key in self._text_search_cache:
            return self._text_search_cache[cache_key].copy()

        results = {
            'query': query,
            'search_type': 'text',
            'documents': [],
            'total_results': 0
        }

        query_lower = query.lower().strip()
        query_words = self._word_pattern.findall(query_lower)  # Use precompiled regex

        if not query_words:
            return results
        
        doc_scores = defaultdict(float)
        doc_ranking_details = {}  # Store detailed ranking info for text search
        
        # OPTIMIZATION: Use reusable thread pool for parallel text search processing
        def score_document(doc_item):
            doc_id, doc = doc_item
            title_lower = doc['title'].lower()
            content_lower = doc['content'].lower()

            # OPTIMIZATION: Quick pre-filtering - skip documents that don't contain any query words
            has_any_word = False
            for word in query_words:
                if word in title_lower or word in content_lower:
                    has_any_word = True
                    break

            if not has_any_word:
                return None  # Skip this document entirely

            # === BASIC KEYWORD MATCHING ===
            score = 0
            title_word_matches = 0
            content_word_matches = 0

            # Optimized word matching
            title_words = set(self._word_pattern.findall(title_lower))
            content_words = set(self._word_pattern.findall(content_lower))

            for word in query_words:
                if word in title_words:
                    score += 3
                    title_word_matches += 1
                if word in content_words:
                    score += 1
                    content_word_matches += 1

            # Exact phrase match gets bonus
            if query_lower in title_lower:
                score += 5
            if query_lower in content_lower:
                score += 2

            if score > 0:
                # === DETAILED RANKING WITH BREAKDOWN ===
                base_score = score
                ranking_details = {
                    'base_score': base_score,
                    'title_matches': title_word_matches,
                    'content_matches': content_word_matches,
                    'query_words': query_words,
                    'search_type': 'text'
                }

                # 1. Basic TF-IDF Score
                tf_idf_score = self._calculate_tf_idf_score(query_words, doc['title'], doc['content'])
                tf_idf_contribution = tf_idf_score * 5
                score += tf_idf_contribution
                ranking_details['tf_idf_score'] = tf_idf_score
                ranking_details['tf_idf_contribution'] = tf_idf_contribution

                # 2. Document quality boost
                doc_quality = self.document_scores.get(doc_id, 1.0)
                quality_multiplier = (1 + doc_quality * 0.05)
                score *= quality_multiplier
                ranking_details['doc_quality'] = doc_quality
                ranking_details['quality_boost'] = quality_multiplier

                # 3. Query coverage bonus
                query_coverage = (title_word_matches + content_word_matches) / len(query_words)
                if query_coverage >= 0.8:
                    coverage_boost = 1.3
                elif query_coverage >= 0.5:
                    coverage_boost = 1.1
                else:
                    coverage_boost = 1.0
                score *= coverage_boost
                ranking_details['query_coverage'] = query_coverage
                ranking_details['coverage_boost'] = coverage_boost

                # 4. Title dominance bonus
                title_boost = 1.2 if title_word_matches > 0 else 1.0
                score *= title_boost
                ranking_details['title_boost'] = title_boost
                ranking_details['final_score'] = score

                return doc_id, score, ranking_details
            return None

        # OPTIMIZATION: Process documents more efficiently
        doc_items = list(self.documents.items())

        # For smaller datasets, process sequentially to avoid thread overhead
        if len(doc_items) < 1000:
            for doc_item in doc_items:
                result = score_document(doc_item)
                if result:
                    doc_id, score, ranking_details = result
                    doc_scores[doc_id] = score
                    doc_ranking_details[doc_id] = ranking_details
        else:
            # For larger datasets, use parallel processing with smaller batches
            batch_size = 500  # Larger batches for better efficiency

            for i in range(0, len(doc_items), batch_size):
                batch = doc_items[i:i + batch_size]
                futures = [self._thread_pool.submit(score_document, item) for item in batch]

                for future in futures:
                    result = future.result()
                    if result:
                        doc_id, score, ranking_details = result
                        doc_scores[doc_id] = score
                        doc_ranking_details[doc_id] = ranking_details
        
        # Use heapq for top-k optimization
        if len(doc_scores) > limit * 2:
            import heapq
            sorted_docs = heapq.nlargest(limit, doc_scores.items(), key=lambda x: x[1])
        else:
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Build result documents
        for doc_id, score in sorted_docs:
            doc = self.documents[doc_id].copy()
            doc['score'] = score
            doc['entity_count'] = len(self.document_entities.get(doc_id, []))
            doc['ranking_details'] = doc_ranking_details.get(doc_id, {})
            results['documents'].append(doc)
        
        results['total_results'] = len(doc_scores)

        # OPTIMIZATION: Cache the results
        if len(self._text_search_cache) >= self._cache_max_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._text_search_cache))
            del self._text_search_cache[oldest_key]

        self._text_search_cache[cache_key] = results.copy()

        return results

    def hybrid_search(self, query: str, limit: int = 10) -> Dict:
        """Combine entity and text search with optimizations"""
        self._ensure_loaded()

        # OPTIMIZATION: Use reusable thread pool for parallel execution of different search types
        def run_entity_search():
            return self.search_by_entity(query, fuzzy=True, limit=limit*2)

        def run_text_search():
            return self.search_text(query, limit=limit*2)

        # Submit both searches to the reusable thread pool
        entity_future = self._thread_pool.submit(run_entity_search)
        text_future = self._thread_pool.submit(run_text_search)

        entity_results = entity_future.result()
        text_results = text_future.result()

        # === ADVANCED HYBRID SCORING ===
        combined_scores = defaultdict(float)
        all_docs = {}
        entity_doc_ids = set()
        text_doc_ids = set()

        # Normalize scores for fair combination
        entity_scores = [doc['score'] for doc in entity_results['documents']]
        text_scores = [doc['score'] for doc in text_results['documents']]

        # Calculate normalization factors
        max_entity_score = max(entity_scores) if entity_scores else 1.0
        max_text_score = max(text_scores) if text_scores else 1.0

        # Add entity search results with normalization
        for doc in entity_results['documents']:
            doc_id = doc['id']
            entity_doc_ids.add(doc_id)

            # Normalize entity score to 0-1 range, then scale
            normalized_entity_score = (doc['score'] / max_entity_score) if max_entity_score > 0 else 0
            combined_scores[doc_id] += normalized_entity_score * 3.0  # Entity weight = 3.0
            all_docs[doc_id] = doc

        # Add text search results with normalization
        for doc in text_results['documents']:
            doc_id = doc['id']
            text_doc_ids.add(doc_id)

            # Normalize text score to 0-1 range, then scale
            normalized_text_score = (doc['score'] / max_text_score) if max_text_score > 0 else 0
            combined_scores[doc_id] += normalized_text_score * 2.0  # Text weight = 2.0

            if doc_id not in all_docs:
                all_docs[doc_id] = doc

        # === HYBRID BONUSES ===
        # Documents that match both entity and text searches get significant bonus
        both_matches = entity_doc_ids.intersection(text_doc_ids)
        for doc_id in both_matches:
            combined_scores[doc_id] *= 1.5  # 50% bonus for dual matches

        # Query-specific bonuses
        query_words = self._word_pattern.findall(query.lower())
        for doc_id, doc in all_docs.items():
            # Bonus for documents where query appears as exact phrase
            if query.lower() in doc['title'].lower():
                combined_scores[doc_id] *= 1.4  # Title phrase match bonus
            elif query.lower() in doc['content'].lower():
                combined_scores[doc_id] *= 1.2  # Content phrase match bonus

            # Bonus for comprehensive query coverage
            doc_text = f"{doc['title']} {doc['content']}".lower()
            doc_words = set(self._word_pattern.findall(doc_text))
            query_coverage = len(set(query_words).intersection(doc_words)) / len(query_words) if query_words else 0

            if query_coverage >= 0.8:
                combined_scores[doc_id] *= 1.3  # High coverage bonus
            elif query_coverage >= 0.6:
                combined_scores[doc_id] *= 1.15  # Medium coverage bonus

        # Use heapq for top-k optimization
        if len(combined_scores) > limit * 2:
            import heapq
            sorted_docs = heapq.nlargest(limit, combined_scores.items(), key=lambda x: x[1])
        else:
            sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        results = {
            'query': query,
            'search_type': 'hybrid',
            'documents': [],
            'total_results': len(combined_scores),
            'entity_matches': entity_results.get('matched_entities', []),
            'text_matches': len(text_results['documents']),
            'dual_matches': len(both_matches),
            'ranking_features': {
                'tf_idf_enabled': True,
                'proximity_scoring': True,
                'entity_context_analysis': True,
                'position_weighting': True,
                'type_relevance': True,
                'hybrid_normalization': True
            },
            'note': 'Advanced ranking with TF-IDF, proximity, and context analysis'
        }

        # Build final results
        for doc_id, score in sorted_docs:
            doc = all_docs[doc_id].copy()
            doc['combined_score'] = score

            # Add hybrid ranking details
            hybrid_ranking = {
                'search_type': 'hybrid',
                'entity_score': 0,
                'text_score': 0,
                'dual_match_bonus': doc_id in both_matches,
                'final_combined_score': score
            }

            # Get original scores from component searches
            if doc_id in entity_doc_ids:
                for entity_doc in entity_results['documents']:
                    if entity_doc['id'] == doc_id:
                        hybrid_ranking['entity_score'] = entity_doc.get('score', 0)
                        hybrid_ranking['entity_ranking'] = entity_doc.get('ranking_details', [])
                        break

            if doc_id in text_doc_ids:
                for text_doc in text_results['documents']:
                    if text_doc['id'] == doc_id:
                        hybrid_ranking['text_score'] = text_doc.get('score', 0)
                        hybrid_ranking['text_ranking'] = text_doc.get('ranking_details', {})
                        break

            doc['ranking_details'] = hybrid_ranking
            results['documents'].append(doc)

        return results
    
    @lru_cache(maxsize=500)
    def get_entity_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get entity suggestions for autocomplete with caching"""
        self._ensure_loaded()
        
        partial_lower = partial_query.lower().strip()
        suggestions = []
        
        # Optimized prefix matching
        for entity_text in self._entity_text_cache:
            if entity_text.startswith(partial_lower):
                suggestions.append(entity_text)
        
        # Sort by frequency
        suggestions.sort(key=lambda x: self.entity_metadata.get(x, {}).get('frequency', 0), reverse=True)
        
        return suggestions[:limit]

    def get_related_entities(self, entity_name: str, limit: int = 5) -> List[Dict]:
        """Get entities that frequently co-occur with the given entity"""
        entity_lower = entity_name.lower().strip()

        if entity_lower not in self.entity_relationships:
            return []

        related = self.entity_relationships[entity_lower]

        # Sort by co-occurrence frequency and add metadata
        related_with_metadata = []
        for related_entity, cooccur_count in related.items():
            if related_entity in self.entity_metadata:
                metadata = self.entity_metadata[related_entity]
                related_with_metadata.append({
                    'entity': related_entity,
                    'type': metadata.get('type', 'UNKNOWN'),
                    'cooccurrence_count': cooccur_count,
                    'frequency': metadata.get('frequency', 1),
                    'avg_quality': metadata.get('avg_doc_quality', 1.0)
                })

        # Use heapq for top-k optimization
        if len(related_with_metadata) > limit * 2:
            import heapq
            return heapq.nlargest(limit, related_with_metadata, key=lambda x: x['cooccurrence_count'])
        else:
            related_with_metadata.sort(key=lambda x: x['cooccurrence_count'], reverse=True)
            return related_with_metadata[:limit]
    
    def get_statistics(self) -> Dict:
        """Get search engine statistics"""
        self._ensure_loaded()
        
        stats = {
            'total_documents': len(self.documents),
            'total_entities': len(self.entity_index),
            'entity_types': {}
        }
        
        for entity_type, entities in self.type_index.items():
            unique_entities = len(set([e['text'] for e in entities]))
            stats['entity_types'][entity_type] = {
                'unique_count': unique_entities,
                'total_mentions': len(entities)
            }
        
        return stats
