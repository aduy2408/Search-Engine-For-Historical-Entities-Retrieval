"""
Demo script for Historical Entity Search Engine
Shows how to use the search functionality
"""

from search_core import HistoricalSearchEngine
import json

def print_results(results, title):
    """Pretty print search results"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Query: {results['query']}")
    print(f"Search Type: {results['search_type']}")
    print(f"Total Results: {results['total_results']}")
    
    if 'matched_entities' in results and results['matched_entities']:
        print(f"Matched Entities: {', '.join(results['matched_entities'])}")
    
    print(f"\nTop {len(results['documents'])} Results:")
    print("-" * 60)
    
    for i, doc in enumerate(results['documents'], 1):
        print(f"\n{i}. {doc['title']}")
        print(f"   Score: {doc.get('score', doc.get('combined_score', 0)):.2f}")
        print(f"   Content: {doc['content'][:150]}...")
        print(f"   Entities: {doc.get('entity_count', 0)}")
        
        if 'matched_entities' in doc:
            entities = [f"{e['entity']}({e['type']})" for e in doc['matched_entities']]
            print(f"   Matched: {', '.join(entities)}")
        
        print(f"   URL: {doc['url']}")

def main():
    """Run search engine demo"""
    print("🏛️ Historical Entity Search Engine Demo")
    print("Loading search indexes...")
    
    # Initialize search engine
    search_engine = HistoricalSearchEngine()
    
    try:
        search_engine.load_indexes()
    except FileNotFoundError:
        print("❌ Search indexes not found!")
        print("Please run: python search_engine/entity_indexer.py")
        return
    
    # Show statistics
    stats = search_engine.get_statistics()
    print(f"\n📊 Database Statistics:")
    print(f"   Documents: {stats['total_documents']}")
    print(f"   Unique Entities: {stats['total_entities']}")
    print(f"   Entity Types:")
    for entity_type, type_stats in stats['entity_types'].items():
        print(f"     {entity_type}: {type_stats['unique_count']} unique, {type_stats['total_mentions']} mentions")
    
    # Demo searches
    demo_queries = [
        ("Dao đỏ", "entity"),
        ("Tây Ninh", "entity"),
        ("LOC", "type"),
        ("lễ hội", "text"),
        ("văn hóa truyền thống", "semantic"),
        ("múa khèn", "hybrid")
    ]
    
    print(f"\n🔍 Running Demo Searches...")
    
    for query, search_type in demo_queries:
        try:
            if search_type == "entity":
                results = search_engine.search_by_entity(query, fuzzy=True, limit=3)
            elif search_type == "type":
                results = search_engine.search_by_type(query, limit=3)
            elif search_type == "text":
                results = search_engine.search_text(query, limit=3)
            elif search_type == "semantic":
                results = search_engine.search_semantic(query, limit=3)
            elif search_type == "hybrid":
                results = search_engine.hybrid_search(query, limit=3)
            
            print_results(results, f"Demo: {search_type.title()} Search")
            
        except Exception as e:
            print(f"❌ Error in {search_type} search for '{query}': {e}")
    
    # Interactive search
    print(f"\n🎯 Interactive Search (type 'quit' to exit)")
    print("Available search types: entity, text, type, hybrid")
    
    while True:
        try:
            query = input("\nEnter search query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            search_type = input("Search type (entity/text/type/semantic/hybrid) [hybrid]: ").strip().lower()
            if not search_type:
                search_type = "hybrid"

            if search_type == "entity":
                results = search_engine.search_by_entity(query, fuzzy=True, limit=5)
            elif search_type == "type":
                results = search_engine.search_by_type(query, limit=5)
            elif search_type == "text":
                results = search_engine.search_text(query, limit=5)
            elif search_type == "semantic":
                results = search_engine.search_semantic(query, limit=5)
            elif search_type == "hybrid":
                results = search_engine.hybrid_search(query, limit=5)
            else:
                print("Invalid search type. Using hybrid search.")
                results = search_engine.hybrid_search(query, limit=5)
            
            print_results(results, "Interactive Search")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    print("\n👋 Demo completed!")

if __name__ == "__main__":
    main()
