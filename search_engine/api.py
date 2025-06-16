"""
Web API for Historical Entity Search Engine
Provides REST endpoints for search functionality with performance optimizations
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging
from search_core import HistoricalSearchEngine
from functools import lru_cache
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize search engine with thread-safe initialization
search_engine = None
_init_lock = threading.Lock()

def get_search_engine():
    """Thread-safe singleton search engine initialization"""
    global search_engine
    if search_engine is None:
        with _init_lock:
            if search_engine is None:
                logger.info("Initializing search engine...")
                search_engine = HistoricalSearchEngine()
                search_engine.load_indexes()
                logger.info("Search engine initialized successfully")
    return search_engine

# Cache for frequent requests
@lru_cache(maxsize=1000)
def cached_search(query: str, search_type: str, limit: int) -> str:
    """Cached search function that returns JSON string"""
    import json
    
    engine = get_search_engine()
    
    # Perform search based on type
    if search_type == 'entity':
        results = engine.search_by_entity(query, fuzzy=True, limit=limit)
    elif search_type == 'text':
        results = engine.search_text(query, limit=limit)
    elif search_type == 'type':
        results = engine.search_by_type(query, limit=limit)
    elif search_type == 'hybrid':
        results = engine.hybrid_search(query, limit=limit)
    else:
        raise ValueError(f'Invalid search type: {search_type}')
    
    return json.dumps(results, ensure_ascii=False)

@lru_cache(maxsize=500)
def cached_suggestions(query: str, limit: int) -> str:
    """Cached suggestions function"""
    import json
    
    engine = get_search_engine()
    suggestions = engine.get_entity_suggestions(query, limit=limit)
    return json.dumps(suggestions, ensure_ascii=False)

@lru_cache(maxsize=100)
def cached_stats() -> str:
    """Cached statistics function"""
    import json
    
    engine = get_search_engine()
    stats = engine.get_statistics()
    return json.dumps(stats, ensure_ascii=False)

# Simple HTML template for testing
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Historical Entity Search Engine</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .search-box { margin-bottom: 30px; text-align: center; }
        input[type="text"] { width: 60%; padding: 12px; font-size: 16px; border: 2px solid #ddd; border-radius: 5px; }
        button { padding: 12px 24px; font-size: 16px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; margin-left: 10px; }
        button:hover { background: #2980b9; }
        .search-type { margin: 20px 0; text-align: center; }
        .search-type label { margin: 0 15px; }
        .results { margin-top: 30px; }
        .result-item { border: 1px solid #ddd; margin: 15px 0; padding: 20px; border-radius: 5px; background: #fafafa; }
        .result-title { font-weight: bold; color: #2c3e50; font-size: 18px; margin-bottom: 10px; }
        .result-content { color: #555; margin-bottom: 10px; }
        .result-meta { font-size: 12px; color: #888; }
        .ranking-score {
            background: #3498db;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 11px;
            margin-right: 10px;
            display: inline-block;
        }
        .ranking-score.high { background: #27ae60; }
        .ranking-score.medium { background: #f39c12; }
        .ranking-score.low { background: #e74c3c; }
        .entity-tag { background: #e74c3c; color: white; padding: 2px 6px; border-radius: 3px; margin: 2px; font-size: 11px; }
        .entity-tag.PER { background: #e74c3c; }
        .entity-tag.LOC { background: #27ae60; }
        .entity-tag.ORG { background: #f39c12; }
        .stats { background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .loading { text-align: center; color: #666; }
        .error { color: #e74c3c; text-align: center; padding: 20px; }
        .performance-info { font-size: 12px; color: #666; margin-top: 10px; text-align: center; }
        .ranking-details {
            font-size: 10px;
            color: #666;
            margin-top: 5px;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 3px;
            border-left: 3px solid #3498db;
            display: none;
        }
        .ranking-toggle {
            cursor: pointer;
            color: #3498db;
            text-decoration: underline;
            font-size: 10px;
            margin-left: 10px;
        }
        .ranking-toggle:hover {
            color: #2980b9;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏛️ Historical Entity Search Engine</h1>
        
        <div class="stats" id="stats">
            <strong>Loading statistics...</strong>
        </div>
        
        <div class="search-box">
            <input type="text" id="searchQuery" placeholder="Search for historical entities, people, places, or events..." onkeypress="handleKeyPress(event)">
            <button onclick="performSearch()">Search</button>
        </div>
        
        <div class="search-type">
            <label><input type="radio" name="searchType" value="hybrid" checked> Hybrid Search (Entity + Text)</label>
            <label><input type="radio" name="searchType" value="entity"> Entity Search</label>
            <label><input type="radio" name="searchType" value="text"> Text Search</label>
            <label><input type="radio" name="searchType" value="type"> By Type (PER/LOC/ORG)</label>
        </div>
        
        <div class="results" id="results"></div>
        <div class="performance-info" id="performanceInfo"></div>
    </div>

    <script>
        // Load statistics on page load
        window.onload = function() {
            loadStatistics();
        };
        
        function loadStatistics() {
            const startTime = performance.now();
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    const loadTime = (performance.now() - startTime).toFixed(2);
                    let statsHtml = '<strong>Database Statistics:</strong> ';
                    statsHtml += `${data.total_documents} documents, ${data.total_entities} unique entities. `;
                    statsHtml += 'Entity types: ';
                    for (let [type, stats] of Object.entries(data.entity_types)) {
                        statsHtml += `${type}(${stats.unique_count}) `;
                    }
                    statsHtml += `<span style="color: #27ae60; margin-left: 15px;">Loaded in ${loadTime}ms</span>`;
                    document.getElementById('stats').innerHTML = statsHtml;
                })
                .catch(error => {
                    document.getElementById('stats').innerHTML = '<span class="error">Failed to load statistics</span>';
                });
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                performSearch();
            }
        }
        
        function performSearch() {
            const query = document.getElementById('searchQuery').value.trim();
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            const searchType = document.querySelector('input[name="searchType"]:checked').value;
            const startTime = performance.now();
            
            document.getElementById('results').innerHTML = '<div class="loading">Searching...</div>';
            document.getElementById('performanceInfo').innerHTML = '';
            
            let endpoint = '/api/search';
            let params = new URLSearchParams({
                q: query,
                type: searchType,
                limit: 20
            });
            
            fetch(`${endpoint}?${params}`)
                .then(response => response.json())
                .then(data => {
                    const searchTime = (performance.now() - startTime).toFixed(2);
                    displayResults(data);
                    document.getElementById('performanceInfo').innerHTML = 
                        `Search completed in ${searchTime}ms (${data.total_results} total results)`;
                })
                .catch(error => {
                    document.getElementById('results').innerHTML = '<div class="error">Search failed: ' + error.message + '</div>';
                });
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            if (data.documents.length === 0) {
                resultsDiv.innerHTML = '<div class="error">No results found</div>';
                return;
            }
            
            let html = `<h3>Found ${data.total_results} results (showing ${data.documents.length})</h3>`;
            
            if (data.matched_entities && data.matched_entities.length > 0) {
                html += `<p><strong>Matched entities:</strong> ${data.matched_entities.join(', ')}</p>`;
            }
            
            data.documents.forEach(doc => {
                html += '<div class="result-item">';
                html += `<div class="result-title">${doc.title}</div>`;
                html += `<div class="result-content">${doc.content.substring(0, 300)}...</div>`;
                
                // Show matched entities if available
                if (doc.matched_entities) {
                    html += '<div>';
                    doc.matched_entities.forEach(entity => {
                        html += `<span class="entity-tag ${entity.type}">${entity.entity} (${entity.type})</span>`;
                    });
                    html += '</div>';
                }
                
                html += `<div class="result-meta">`;

                // Display ranking score with color coding
                const score = doc.score || doc.combined_score || 0;
                let scoreClass = 'low';
                if (score > 10) scoreClass = 'high';
                else if (score > 5) scoreClass = 'medium';

                html += `<span class="ranking-score ${scoreClass}">Score: ${score.toFixed(2)}</span>`;
                html += `Entities: ${doc.entity_count} | `;
                html += `<a href="${doc.url}" target="_blank">View Source</a>`;

                // Add ranking details toggle
                if (doc.ranking_details) {
                    html += `<span class="ranking-toggle" onclick="toggleRankingDetails('${doc.id}')">Show Ranking Details</span>`;
                }

                html += `</div>`;

                // Add detailed ranking information (hidden by default)
                if (doc.ranking_details) {
                    html += `<div class="ranking-details" id="ranking-${doc.id}">`;
                    html += formatRankingDetails(doc.ranking_details, data.search_type);
                    html += `</div>`;
                }
                html += '</div>';
            });
            
            resultsDiv.innerHTML = html;
        }

        function toggleRankingDetails(docId) {
            const detailsDiv = document.getElementById(`ranking-${docId}`);
            const toggle = event.target;

            if (detailsDiv.style.display === 'none' || detailsDiv.style.display === '') {
                detailsDiv.style.display = 'block';
                toggle.textContent = 'Hide Ranking Details';
            } else {
                detailsDiv.style.display = 'none';
                toggle.textContent = 'Show Ranking Details';
            }
        }

        function formatRankingDetails(rankingDetails, searchType) {
            let html = '<strong>🎯 Ranking Breakdown:</strong><br>';

            if (searchType === 'entity') {
                // Format entity search ranking details
                if (Array.isArray(rankingDetails)) {
                    rankingDetails.forEach((detail, index) => {
                        html += `<div style="margin: 5px 0; padding: 3px; border-left: 2px solid #e74c3c;">`;
                        html += `<strong>Entity ${index + 1}:</strong> ${detail.entity_name || 'N/A'}<br>`;
                        html += `• Base Score: ${(detail.base_score || 0).toFixed(2)} (freq: ${detail.entity_frequency || 0}, similarity: ${(detail.similarity || 0).toFixed(2)})<br>`;
                        html += `• Context Relevance: ${(detail.context_relevance || 1).toFixed(2)}x<br>`;
                        html += `• Quality Boost: ${(detail.quality_boost || 1).toFixed(2)}x<br>`;
                        html += `• Position Score: ${(detail.position_score || 1).toFixed(2)}x<br>`;
                        if (detail.in_title) html += `• Title Boost: ${(detail.title_boost || 1).toFixed(2)}x<br>`;
                        html += `• Final Score: ${(detail.final_score || 0).toFixed(2)}<br>`;
                        html += `</div>`;
                    });
                }
            } else if (searchType === 'text') {
                // Format text search ranking details
                html += `• Base Score: ${(rankingDetails.base_score || 0).toFixed(2)}<br>`;
                html += `• Title Matches: ${rankingDetails.title_matches || 0}, Content Matches: ${rankingDetails.content_matches || 0}<br>`;
                html += `• TF-IDF Contribution: ${(rankingDetails.tf_idf_contribution || 0).toFixed(2)}<br>`;
                html += `• Query Coverage: ${((rankingDetails.query_coverage || 0) * 100).toFixed(1)}% (boost: ${(rankingDetails.coverage_boost || 1).toFixed(2)}x)<br>`;
                html += `• Quality Boost: ${(rankingDetails.quality_boost || 1).toFixed(2)}x<br>`;
                html += `• Title Boost: ${(rankingDetails.title_boost || 1).toFixed(2)}x<br>`;
                html += `• Final Score: ${(rankingDetails.final_score || 0).toFixed(2)}<br>`;
            } else if (searchType === 'hybrid') {
                // Format hybrid search ranking details
                html += `• Entity Score: ${(rankingDetails.entity_score || 0).toFixed(2)}<br>`;
                html += `• Text Score: ${(rankingDetails.text_score || 0).toFixed(2)}<br>`;
                html += `• Dual Match Bonus: ${rankingDetails.dual_match_bonus ? 'Yes (1.5x)' : 'No'}<br>`;
                html += `• Combined Score: ${(rankingDetails.final_combined_score || 0).toFixed(2)}<br>`;
            }

            return html;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main search interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/search')
def search():
    """Main search endpoint with caching and optimization"""
    try:
        query = request.args.get('q', '').strip()
        search_type = request.args.get('type', 'hybrid')
        limit = min(int(request.args.get('limit', 10)), 100)  # Cap limit to prevent abuse
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        if len(query) > 500:  # Prevent extremely long queries
            return jsonify({'error': 'Query too long'}), 400
        
        # Use cached search for better performance
        start_time = time.time()
        
        try:
            results_json = cached_search(query, search_type, limit)
            import json
            results_dict = json.loads(results_json)  # Safely convert JSON string back to dict
            results = jsonify(results_dict)
            
            # Add performance metrics
            search_time = (time.time() - start_time) * 1000
            logger.info(f"Search completed in {search_time:.2f}ms for query: '{query[:50]}...'")
            
            return results
            
        except ValueError as e:
            if 'semantic search' in str(e).lower():
                return jsonify({'error': 'Semantic search has been moved to semantic_search/ folder'}), 400
            return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggest')
def suggest():
    """Entity suggestion endpoint for autocomplete with caching"""
    try:
        query = request.args.get('q', '').strip()
        limit = min(int(request.args.get('limit', 10)), 50)  # Cap limit
        
        if not query:
            return jsonify([])
        
        if len(query) > 100:  # Prevent long suggestion queries
            return jsonify([])
        
        # Use cached suggestions
        suggestions_json = cached_suggestions(query, limit)
        import json
        suggestions = json.loads(suggestions_json)
        
        return jsonify(suggestions)
    
    except Exception as e:
        logger.error(f"Suggestion error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    """Get search engine statistics with caching"""
    try:
        # Use cached statistics
        stats_json = cached_stats()
        import json
        statistics = json.loads(stats_json)
        
        return jsonify(statistics)
    
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/entity/<entity_name>')
def entity_details(entity_name):
    """Get details about a specific entity with caching"""
    try:
        # Cache entity details
        @lru_cache(maxsize=200)
        def get_cached_entity_details(name: str):
            engine = get_search_engine()
            results = engine.search_by_entity(name, fuzzy=False, limit=50)
            
            # Add related entities
            related = engine.get_related_entities(name, limit=10)
            results['related_entities'] = related
            
            return results
        
        results = get_cached_entity_details(entity_name)
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Entity details error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/related/<entity_name>')
def related_entities(entity_name):
    """Get related entities with caching"""
    try:
        limit = min(int(request.args.get('limit', 5)), 20)
        
        @lru_cache(maxsize=200)
        def get_cached_related(name: str, lim: int):
            engine = get_search_engine()
            return engine.get_related_entities(name, limit=lim)
        
        related = get_cached_related(entity_name, limit)
        return jsonify(related)
    
    except Exception as e:
        logger.error(f"Related entities error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        engine = get_search_engine()
        return jsonify({
            'status': 'healthy',
            'loaded': engine.loaded,
            'cache_info': {
                'search_cache_size': cached_search.cache_info().currsize,
                'suggestions_cache_size': cached_suggestions.cache_info().currsize,
                'stats_cache_size': cached_stats.cache_info().currsize
            }
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/cache/clear')
def clear_cache():
    """Clear API caches"""
    try:
        cached_search.cache_clear()
        cached_suggestions.cache_clear()
        cached_stats.cache_clear()
        
        return jsonify({'message': 'Caches cleared successfully'})
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return jsonify({'error': str(e)}), 500

def initialize_search_engine():
    """Initialize search engine on startup"""
    try:
        logger.info("Initializing search engine on startup...")
        get_search_engine()
        logger.info("Search engine initialization completed")
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        raise

# Initialize search engine when module is imported
try:
    # Use a separate thread for initialization to avoid blocking
    init_thread = threading.Thread(target=initialize_search_engine)
    init_thread.daemon = True
    init_thread.start()
except Exception as e:
    logger.error(f"Failed to start initialization thread: {e}")

if __name__ == '__main__':
    # Ensure search engine is initialized before starting server
    get_search_engine()
    
    # Run with optimized settings
    app.run(
        host='0.0.0.0',
        port=5001,  # Changed to avoid port conflicts
        debug=False,  # Disable debug mode for better performance
        threaded=True,  # Enable threading
        processes=1  # Use single process with threading
    )
