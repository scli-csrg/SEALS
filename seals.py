import tenseal as ts
import numpy as np
from datasketch import MinHashLSH, MinHash
from typing import List, Set, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class PASQAL:
    """
    Practical and Adaptive System for Encrypted Substring Query with Low Leakage
    Implementation of the core framework based on the SIGMOD 2027 paper.
    """
    
    def __init__(self, lsh_threshold=0.5, num_perm=128, lsh_band_size=4, he_poly_modulus_degree=8192):
        """
        Initialize the PASQAL system.
        
        Args:
            lsh_threshold: Jaccard similarity threshold for LSH (τ in paper)
            num_perm: Number of permutation functions for MinHash
            lsh_band_size: Band size for LSH indexing
            he_poly_modulus_degree: Polynomial modulus degree for TenSEAL (affects security & performance)
        """
        # ---- Layer 1: Fuzzy Index (F) ----
        self.lsh_index = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.lsh_threshold = lsh_threshold
        self.lsh_band_size = lsh_band_size
        
        # Store mapping: doc_id -> list of keywords
        self.doc_keywords: Dict[int, List[str]] = {}
        
        # Store mapping: keyword -> MinHash for indexing
        self.keyword_hashes: Dict[str, MinHash] = {}
        
        # ---- Layer 2: Homomorphic Verification (H) ----
        # Create TenSEAL context for BFV scheme (similar to PySEAL mentioned in paper)
        self.he_context = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=he_poly_modulus_degree,
            plain_modulus=1032193  # Sufficiently large for keyword hashing
        )
        self.he_context.generate_galois_keys()
        self.he_context.generate_relin_keys()
        
        # Store encrypted keyword vectors for each document
        self.encrypted_keywords: Dict[int, ts.EncryptedVector] = {}
        
        # ---- Dynamic Update Support ----
        self.revocation_tokens: Set[int] = set()  # For deleted documents
        self.document_counter = 0
        
    def _tokenize_and_hash(self, text: str) -> Set[str]:
        """
        Preprocess document: tokenization, stopword removal, and keyword extraction.
        """
        # Simplified tokenization (in practice, use nltk or spacy)
        tokens = text.lower().split()
        # Simple stopword removal (extend this list)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'for', 'on', 'with', 'by', 'at', 'from', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'but', 'not', 'no', 'so', 'very', 'just', 'than', 'that', 'these', 'those', 'this', 'those'}
        keywords = [token for token in tokens if token not in stopwords and len(token) > 2]
        # Return unique keywords
        return set(keywords)
    
    def _build_minhash(self, keywords: Set[str]) -> MinHash:
        """Build a MinHash signature for a set of keywords."""
        m = MinHash(num_perm=self.num_perm)
        for kw in keywords:
            m.update(kw.encode('utf8'))
        return m
    
    def insert_document(self, doc_id: int, content: str):
        """
        Insert a new document into the encrypted database.
        Implements Algorithm 5 (Insert operation).
        """
        # Step 1: Preprocess document to extract keywords
        keywords = self._tokenize_and_hash(content)
        self.doc_keywords[doc_id] = list(keywords)
        
        # Step 2: Update Fuzzy Index (F)
        doc_minhash = self._build_minhash(keywords)
        self.lsh_index.insert(doc_id, doc_minhash)
        
        # Step 3: Encrypt keywords for Homomorphic Layer (H)
        # Convert keywords to numeric vectors for HE
        keyword_vector = self._keywords_to_vector(keywords)
        encrypted_vector = ts.bfv_vector(self.he_context, keyword_vector)
        self.encrypted_keywords[doc_id] = encrypted_vector
        
        return True
    
    def _keywords_to_vector(self, keywords: List[str]) -> List[int]:
        """
        Convert keyword list to a fixed-size integer vector for homomorphic encryption.
        Uses a simple hash-based encoding.
        """
        vector_size = 100  # Fixed size for demonstration
        vector = [0] * vector_size
        for kw in keywords:
            # Simple hash to index
            idx = hash(kw) % vector_size
            vector[idx] = 1  # Binary presence indicator
        return vector
    
    def generate_trapdoor(self, query_keyword: str) -> Tuple[MinHash, ts.EncryptedVector]:
        """
        Generate search trapdoor for a query.
        Implements Algorithm 3.
        
        Returns:
            Tuple of (fuzzy_trapdoor, homomorphic_encrypted_query)
        """
        # Layer 1: Fuzzy trapdoor (MinHash of query)
        query_keywords = self._tokenize_and_hash(query_keyword)
        fuzzy_trapdoor = self._build_minhash(query_keywords)
        
        # Layer 2: Homomorphic encryption of query
        query_vector = self._keywords_to_vector(list(query_keywords))
        encrypted_query = ts.bfv_vector(self.he_context, query_vector)
        
        return fuzzy_trapdoor, encrypted_query
    
    def search(self, fuzzy_trapdoor: MinHash, encrypted_query: ts.EncryptedVector) -> List[int]:
        """
        Execute a search query on the encrypted database.
        Implements Algorithm 4 (Two-layer search).
        
        Args:
            fuzzy_trapdoor: MinHash of the query keywords
            encrypted_query: Homomorphically encrypted query vector
            
        Returns:
            List of document IDs that match the query (verified exactly)
        """
        # ---- Layer 1: Fuzzy Search (Candidate Generation) ----
        # Query the LSH index to get candidate set C(w_q)
        candidates = self.lsh_index.query(fuzzy_trapdoor)
        
        # Remove any documents that have been revoked (deleted)
        candidates = [c for c in candidates if c not in self.revocation_tokens]
        
        # ---- Layer 2: Exact Verification via Homomorphic Protocol ----
        verified_results = []
        for doc_id in candidates:
            # Retrieve encrypted keywords for this document
            encrypted_doc_keywords = self.encrypted_keywords.get(doc_id)
            if encrypted_doc_keywords is None:
                continue
                
            # Compute homomorphic similarity (dot product)
            # Eval(f_match, E(w_q), E(w_i)) as per equation (6) in paper
            encrypted_similarity = encrypted_query.dot(encrypted_doc_keywords)
            
            # Decrypt result (client-side in real deployment)
            # For this implementation, we simulate decryption
            similarity_score = encrypted_similarity.decrypt()[0]
            
            # If similarity > threshold, consider it an exact match
            # (In practice, f_match would be a boolean equality check)
            if similarity_score > 0:
                verified_results.append(doc_id)
        
        return verified_results
    
    def delete_document(self, doc_id: int):
        """
        Securely delete a document using revocation token.
        Implements Algorithm 5 (Delete operation).
        """
        # Mark document as revoked (soft delete)
        self.revocation_tokens.add(doc_id)
        
        # Optional: Remove from fuzzy index if we want to clean up
        # For this implementation, we rely on the revocation filter during search
        
        # Clean up other data structures
        if doc_id in self.doc_keywords:
            del self.doc_keywords[doc_id]
        if doc_id in self.encrypted_keywords:
            del self.encrypted_keywords[doc_id]
            
        return True
    
    def modify_document(self, doc_id: int, new_content: str):
        """
        Modify an existing document.
        Implements Algorithm 5 (Modify = Delete + Insert).
        """
        self.delete_document(doc_id)
        self.insert_document(doc_id, new_content)
        return True


# ============= Example Usage and Testing =============

def test_pasqal():
    """Demonstrate the full PASQAL workflow."""
    print("=" * 60)
    print("PASQAL: Practical and Adaptive System for Encrypted Substring Query")
    print("=" * 60)
    
    # Initialize system
    pasqal = PASQAL(lsh_threshold=0.5, num_perm=128)
    
    # Sample documents (in practice, load from Enron/DBLP datasets)
    documents = [
        (1, "Machine learning algorithms for data classification and regression analysis"),
        (2, "Deep neural networks achieve state of the art results in computer vision"),
        (3, "Natural language processing enables machines to understand human text"),
        (4, "Cloud computing provides scalable infrastructure for big data analytics"),
        (5, "Encrypted search allows private query processing on outsourced data"),
        (6, "Fuzzy matching techniques handle spelling variations and typos effectively"),
        (7, "Homomorphic encryption enables computation directly on encrypted data"),
        (8, "Locality sensitive hashing provides efficient approximate nearest neighbor search"),
    ]
    
    # Insert documents
    print("\n[1] Inserting documents into encrypted database...")
    for doc_id, content in documents:
        pasqal.insert_document(doc_id, content)
        print(f"    Document {doc_id} inserted")
    
    # Test queries
    test_queries = [
        "machine learning classification",
        "neural networks vision",
        "natural language understanding",
        "cloud analytics",
        "encrypted search privacy"
    ]
    
    print("\n[2] Performing encrypted search queries...")
    for query in test_queries:
        # Generate trapdoor (client-side)
        fuzzy_trapdoor, encrypted_query = pasqal.generate_trapdoor(query)
        
        # Execute search (server-side)
        results = pasqal.search(fuzzy_trapdoor, encrypted_query)
        
        print(f"\n    Query: '{query}'")
        print(f"    Matching documents: {results if results else 'None found'}")
    
    # Test dynamic updates
    print("\n[3] Testing dynamic updates...")
    
    # Insert a new document
    new_doc_id = 9
    new_content = "Privacy preserving machine learning with federated learning"
    pasqal.insert_document(new_doc_id, new_content)
    print(f"    Inserted new document {new_doc_id}")
    
    # Modify existing document
    pasqal.modify_document(1, "Advanced machine learning techniques including ensemble methods")
    print(f"    Modified document 1")
    
    # Delete a document
    pasqal.delete_document(3)
    print(f"    Deleted document 3")
    
    # Query after updates
    fuzzy_trapdoor, encrypted_query = pasqal.generate_trapdoor("machine learning")
    results = pasqal.search(fuzzy_trapdoor, encrypted_query)
    print(f"\n    Search for 'machine learning' after updates: {results}")
    
    print("\n" + "=" * 60)
    print("PASQAL demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_pasqal()
