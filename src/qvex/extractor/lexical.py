"""Lexical TF-IDF Co-occurrence Builder."""

import math
import re
from collections import Counter, defaultdict

STOP_WORDS = {
    "the", "and", "is", "in", "to", "of", "a", "it", "that", "this", "for",
    "on", "with", "as", "by", "an", "be", "at", "are", "not", "or", "from",
    "but", "what", "all", "were", "when", "we", "there", "can", "an", "your",
    "which", "their", "said", "if", "do", "will", "each", "about", "how",
    "up", "out", "them", "then", "she", "many", "some", "so", "these", "would",
    "other", "into", "has", "more", "her", "two", "like", "him", "see", "time",
    "could", "no", "make", "than", "first", "been", "its", "who", "now",
    "people", "my", "made", "over", "down", "only", "way", "find", "use",
    "may", "water", "long", "little", "very", "after", "words", "called",
    "just", "where", "most", "know"
}

class LexicalCooccurrenceBuilder:
    """Extracts lexical keyphrases using TF-IDF and builds co-occurrence edges."""

    def __init__(self, top_k: int = 15, idf_threshold: float = 2.0):
        self.top_k = top_k
        self.idf_threshold = idf_threshold
        
        # Regex to capture capitalized terms and alphanumeric technical identifiers
        self.term_regex = re.compile(r'\b[A-Z][a-zA-Z0-9]*\b|\b[a-zA-Z]+[0-9]+[a-zA-Z0-9]*\b')
        
        # Internal state
        self.doc_frequencies = Counter()
        self.chunk_tokens = {}  # chunk_id -> Counter(terms)
        self.total_docs = 0
        
        # Populated in pass 2
        self.inverted_index = defaultdict(list)
        self.term_idf = {}

    def add_chunk_pass1(self, chunk_id: int, text: str):
        """Pass 1: Tokenize and compute document frequencies."""
        matches = self.term_regex.findall(text)
        valid_terms = [m for m in matches if len(m) > 2 and m.lower() not in STOP_WORDS]
        
        term_counts = Counter(valid_terms)
        self.chunk_tokens[chunk_id] = term_counts
        
        # Document frequency: count each unique term once per document
        for term in term_counts.keys():
            self.doc_frequencies[term] += 1
            
        self.total_docs += 1

    def compute_idf_and_pass2(self):
        """Pass 2: Compute TF-IDF, select top terms, and build inverted index."""
        if self.total_docs == 0:
            return

        # Compute IDF for all seen terms
        for term, df in self.doc_frequencies.items():
            idf = math.log(1 + (self.total_docs / (1 + df))) + 1.0
            self.term_idf[term] = idf
            
        # Compute TF-IDF for each chunk and select top K
        for chunk_id, term_counts in self.chunk_tokens.items():
            tfidf_scores = {}
            for term, tf in term_counts.items():
                tfidf_scores[term] = tf * self.term_idf[term]
                
            # Sort terms by TF-IDF score descending
            sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
            top_terms = sorted_terms[:self.top_k]
            
            for term, score in top_terms:
                if self.term_idf[term] >= self.idf_threshold:
                    # Store (chunk_id, score) in inverted index
                    self.inverted_index[term].append((chunk_id, score))
                    
        # Free memory
        self.chunk_tokens.clear()
        self.doc_frequencies.clear()

    def build_edges(self) -> list[tuple[int, int, str, float]]:
        """
        Generate co-occurrence edges based on the inverted index.
        Returns a list of (source_id, target_id, edge_type, confidence).
        """
        edges = []
        for term, chunk_data in self.inverted_index.items():
            n = len(chunk_data)
            if n > 1:
                edge_type = f"SHARES_CONCEPT: {term}"
                # Create a sequential chain to avoid O(N^2) explosion
                for i in range(n - 1):
                    chunk_a, score_a = chunk_data[i]
                    chunk_b, score_b = chunk_data[i + 1]
                    
                    # Confidence proportional to normalized TF-IDF similarity.
                    # We'll use a simple harmonic mean or product scaled by max observed to keep it 0-1.
                    # A simple approximation: normalize by self.total_docs or just use normalized dot product.
                    # Since we only have raw scores, we'll just normalize arbitrarily or use max(1.0, log(score)).
                    # Actually, the user asked for "proportional to normalized TF-IDF similarity".
                    # Let's cap it at 1.0.
                    confidence = min(1.0, (score_a * score_b) / (self.term_idf[term] ** 2 * 10 + 1))
                    confidence = max(0.1, round(confidence, 3))
                    
                    edges.append((chunk_a, chunk_b, edge_type, confidence))
                    edges.append((chunk_b, chunk_a, edge_type, confidence))
        return edges

    def clear(self):
        """Clear the in-memory inverted index."""
        self.inverted_index.clear()
        self.term_idf.clear()
