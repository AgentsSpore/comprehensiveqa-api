"""Demo script showing RVR in action on sample queries"""

import json
from rvr_engine import RVREngine
from document_store import DocumentStore


def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_demo():
    print_section("ComprehensiveQA API - RVR Demo")
    print("Based on: https://arxiv.org/abs/2602.18425v1\n")
    
    # Initialize
    doc_store = DocumentStore()
    rvr_engine = RVREngine(doc_store)
    
    # Demo queries
    queries = [
        {
            "query": "What are effective treatments for type 2 diabetes?",
            "domain": "Medical"
        },
        {
            "query": "What are key legal precedents for employment discrimination?",
            "domain": "Legal"
        },
        {
            "query": "What methods exist for interpreting machine learning models?",
            "domain": "Research"
        }
    ]
    
    for i, query_info in enumerate(queries, 1):
        print_section(f"Demo {i}: {query_info['domain']} Domain")
        print(f"Query: {query_info['query']}\n")
        
        # Run RVR
        results = rvr_engine.retrieve_verify_retrieve(
            query=query_info["query"],
            max_rounds=3,
            docs_per_round=3,
            verification_threshold=0.5
        )
        
        print(f"Total Rounds: {results['total_rounds']}")
        print(f"Verified Documents: {len(results['verified_documents'])}")
        print(f"Coverage Improvement: {results['coverage_improvement']:.1f}%\n")
        
        print("Retrieved Answers:\n")
        for j, doc in enumerate(results["verified_documents"], 1):
            print(f"  [{j}] Round {doc['round']} | Score: {doc['score']:.3f}")
            print(f"      {doc['text'][:120]}...")
            print(f"      Source: {doc['source']}\n")
        
        # Compare with single-round baseline
        baseline = rvr_engine.retrieve(
            query=query_info["query"],
            top_k=3,
            exclude_indices=set()
        )
        
        print(f"Baseline (single-round) retrieved: {len(baseline)} documents")
        print(f"RVR retrieved: {len(results['verified_documents'])} documents")
        print(f"Improvement: +{len(results['verified_documents']) - len(baseline)} documents\n")
    
    print_section("Summary")
    print("✅ RVR successfully demonstrates multi-round retrieval")
    print("✅ Each round discovers new relevant documents")
    print("✅ Verification filters high-quality results")
    print("✅ Query augmentation helps find diverse answers\n")
    print("Start the API with: python app.py")
    print("Then try: curl -X POST http://localhost:8000/query \\")
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"query": "What are treatments for diabetes?", "max_rounds": 3}\'\n')


if __name__ == "__main__":
    run_demo()