"""RVR Engine: Implements Retrieve-Verify-Retrieve algorithm"""

import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RVREngine:
    """Multi-round retrieval with verification for comprehensive QA"""
    
    def __init__(self, document_store):
        self.doc_store = document_store
        # Use lightweight model for MVP
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_embeddings = None
        self._encode_documents()
    
    def _encode_documents(self):
        """Pre-compute document embeddings"""
        texts = [doc["text"] for doc in self.doc_store.documents]
        self.doc_embeddings = self.encoder.encode(texts, convert_to_tensor=False)
    
    def retrieve(self, query: str, top_k: int, exclude_indices: set) -> List[Dict]:
        """Retrieve top-k documents for query, excluding already seen ones"""
        query_emb = self.encoder.encode([query], convert_to_tensor=False)
        similarities = cosine_similarity(query_emb, self.doc_embeddings)[0]
        
        # Mask excluded documents
        for idx in exclude_indices:
            similarities[idx] = -1.0
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include valid similarities
                results.append({
                    "index": int(idx),
                    "text": self.doc_store.documents[idx]["text"],
                    "source": self.doc_store.documents[idx]["source"],
                    "category": self.doc_store.documents[idx]["category"],
                    "score": float(similarities[idx])
                })
        
        return results
    
    def verify(self, query: str, documents: List[Dict], threshold: float) -> List[Dict]:
        """Verify document relevance and quality"""
        verified = []
        
        for doc in documents:
            # Re-score with query for verification
            query_emb = self.encoder.encode([query], convert_to_tensor=False)
            doc_emb = self.encoder.encode([doc["text"]], convert_to_tensor=False)
            relevance_score = float(cosine_similarity(query_emb, doc_emb)[0][0])
            
            # Verification: check if above threshold
            if relevance_score >= threshold:
                doc["verified_score"] = relevance_score
                verified.append(doc)
        
        return verified
    
    def augment_query(self, original_query: str, verified_docs: List[Dict]) -> str:
        """Augment query with verified document context"""
        if not verified_docs:
            return original_query
        
        # Extract key information from verified docs
        contexts = [doc["text"][:100] for doc in verified_docs[-2:]]  # Last 2 docs
        augmented = f"{original_query} Context from previous answers: {' '.join(contexts)}"
        
        return augmented
    
    def retrieve_verify_retrieve(
        self, 
        query: str, 
        max_rounds: int = 3,
        docs_per_round: int = 5,
        verification_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Execute RVR algorithm"""
        
        all_verified = []
        seen_indices = set()
        current_query = query
        
        for round_num in range(1, max_rounds + 1):
            # RETRIEVE: Get candidate documents
            candidates = self.retrieve(
                query=current_query,
                top_k=docs_per_round,
                exclude_indices=seen_indices
            )
            
            if not candidates:
                break  # No more documents to retrieve
            
            # VERIFY: Filter high-quality documents
            verified = self.verify(
                query=query,  # Use original query for verification
                documents=candidates,
                threshold=verification_threshold
            )
            
            # Track verified documents
            for doc in verified:
                doc["round"] = round_num
                all_verified.append(doc)
                seen_indices.add(doc["index"])
            
            # RETRIEVE (next round): Augment query with verified context
            if round_num < max_rounds:
                current_query = self.augment_query(query, all_verified)
        
        # Calculate coverage improvement (simulated metric)
        baseline_coverage = docs_per_round
        rvr_coverage = len(all_verified)
        coverage_improvement = ((rvr_coverage - baseline_coverage) / baseline_coverage * 100) if baseline_coverage > 0 else 0
        
        return {
            "total_rounds": min(round_num, max_rounds),
            "verified_documents": [
                {
                    "text": doc["text"],
                    "score": doc["verified_score"],
                    "round": doc["round"],
                    "source": doc["source"]
                }
                for doc in all_verified
            ],
            "coverage_improvement": coverage_improvement
        }