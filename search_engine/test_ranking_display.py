#!/usr/bin/env python3
"""
Test Ranking Display - Test the detailed ranking information display
"""

import sys
import os
import json

# Add the search_engine directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search_core import HistoricalSearchEngine

def test_ranking_details():
    """Test the detailed ranking information"""
    print("🎯 Testing Detailed Ranking Information")
    print("=" * 50)
    
    # Initialize search engine
    print("Loading search engine...")
    engine = HistoricalSearchEngine()
    engine._ensure_loaded()
    print("✅ Search engine loaded")
    print()
    
    # Test different search types
    test_cases = [
        ("Hồ Chí Minh", "entity"),
        ("chiến tranh Việt Nam", "text"),
        ("Nguyễn Ái Quốc", "hybrid"),
    ]
    
    for query, search_type in test_cases:
        print(f"🔍 Testing {search_type.upper()} search: '{query}'")
        print("-" * 40)
        
        # Perform search
        if search_type == "entity":
            results = engine.search_by_entity(query, fuzzy=True, limit=3)
        elif search_type == "text":
            results = engine.search_text(query, limit=3)
        elif search_type == "hybrid":
            results = engine.hybrid_search(query, limit=3)
        
        # Display results with ranking details
        if results['documents']:
            for i, doc in enumerate(results['documents'][:2]):  # Show top 2 results
                print(f"\n📄 Result {i+1}: {doc['title'][:60]}...")
                
                # Display main score
                score = doc.get('score', doc.get('combined_score', 0))
                print(f"   🎯 Final Score: {score:.2f}")
                
                # Display ranking details
                if 'ranking_details' in doc and doc['ranking_details']:
                    print("   📊 Ranking Breakdown:")
                    
                    if search_type == "entity" and isinstance(doc['ranking_details'], list):
                        for j, detail in enumerate(doc['ranking_details']):
                            print(f"      Entity {j+1}: {detail.get('entity_name', 'N/A')}")
                            print(f"        • Base Score: {detail.get('base_score', 0):.2f}")
                            print(f"        • Similarity: {detail.get('similarity', 0):.2f}")
                            print(f"        • Context Relevance: {detail.get('context_relevance', 1):.2f}x")
                            print(f"        • Quality Boost: {detail.get('quality_boost', 1):.2f}x")
                            print(f"        • Position Score: {detail.get('position_score', 1):.2f}x")
                            if detail.get('in_title'):
                                print(f"        • Title Boost: {detail.get('title_boost', 1):.2f}x")
                            print(f"        • Final Score: {detail.get('final_score', 0):.2f}")
                    
                    elif search_type == "text":
                        detail = doc['ranking_details']
                        print(f"      • Base Score: {detail.get('base_score', 0):.2f}")
                        print(f"      • Title Matches: {detail.get('title_matches', 0)}")
                        print(f"      • Content Matches: {detail.get('content_matches', 0)}")
                        print(f"      • TF-IDF Contribution: {detail.get('tf_idf_contribution', 0):.2f}")
                        print(f"      • Query Coverage: {(detail.get('query_coverage', 0) * 100):.1f}%")
                        print(f"      • Coverage Boost: {detail.get('coverage_boost', 1):.2f}x")
                        print(f"      • Quality Boost: {detail.get('quality_boost', 1):.2f}x")
                        print(f"      • Title Boost: {detail.get('title_boost', 1):.2f}x")
                    
                    elif search_type == "hybrid":
                        detail = doc['ranking_details']
                        print(f"      • Entity Score: {detail.get('entity_score', 0):.2f}")
                        print(f"      • Text Score: {detail.get('text_score', 0):.2f}")
                        print(f"      • Dual Match Bonus: {'Yes' if detail.get('dual_match_bonus') else 'No'}")
                        print(f"      • Combined Score: {detail.get('final_combined_score', 0):.2f}")
                else:
                    print("   ⚠️  No ranking details available")
        else:
            print("   ❌ No results found")
        
        print("\n" + "=" * 50)
    
    print("\n🎉 Ranking details test completed!")
    print("\n💡 To see this in the web interface:")
    print("   1. Start the API: python search_engine/api.py")
    print("   2. Open: http://localhost:5001")
    print("   3. Search for any query")
    print("   4. Click 'Show Ranking Details' for detailed breakdown")

def test_score_transparency():
    """Test score transparency and user understanding"""
    print("\n🔍 Testing Score Transparency")
    print("=" * 40)
    
    engine = HistoricalSearchEngine()
    engine._ensure_loaded()
    
    # Test a specific query to show score breakdown
    query = "Hồ Chí Minh"
    results = engine.search_by_entity(query, fuzzy=True, limit=1)
    
    if results['documents']:
        doc = results['documents'][0]
        print(f"Query: '{query}'")
        print(f"Top Result: {doc['title'][:50]}...")
        print(f"Final Score: {doc['score']:.2f}")
        
        if 'ranking_details' in doc and doc['ranking_details']:
            detail = doc['ranking_details'][0]
            print("\n🧮 Score Calculation:")
            print(f"  Base Score = Entity Frequency × Similarity")
            print(f"             = {detail.get('entity_frequency', 0)} × {detail.get('similarity', 0):.2f}")
            print(f"             = {detail.get('base_score', 0):.2f}")
            print()
            print(f"  Final Score = Base Score × Context × Quality × Position × Title × Match × Type")
            print(f"              = {detail.get('base_score', 0):.2f} × {detail.get('context_relevance', 1):.2f} × {detail.get('quality_boost', 1):.2f} × {detail.get('position_score', 1):.2f} × {detail.get('title_boost', 1):.2f} × {detail.get('exact_match_bonus', 1):.2f} × {detail.get('type_relevance', 1):.2f}")
            print(f"              = {detail.get('final_score', 0):.2f}")
            print()
            print("✅ Users can now see exactly how scores are calculated!")

if __name__ == "__main__":
    try:
        test_ranking_details()
        test_score_transparency()
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
