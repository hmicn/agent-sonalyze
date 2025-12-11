import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# -----------------------------
# Settings
# -----------------------------
JSON_PATH = "solutions.json"
TOP_K = 5
TEXT_FIELD = "faiblesses"
ID_FIELD = "id"

def load_data(json_path=JSON_PATH):
    """Loads the JSON data and builds the TF-IDF index."""
    if not os.path.exists(json_path):
        # Fallback for when running from a different directory
        json_path = os.path.join(os.path.dirname(__file__), json_path)
    
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    texts = [doc[TEXT_FIELD] for doc in docs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    return docs, vectorizer, tfidf_matrix

def search_solutions(query, docs, vectorizer, tfidf_matrix, top_k=TOP_K):
    """Searches for solutions based on the query."""
    if not query:
        return []

    # Convert query to TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Cosine similarity
    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()

    # Get top K results
    top_indices = np.argsort(-scores)[:top_k]
    
    results = []
    for idx in top_indices:
        score = scores[idx]
        # Only include relevant results
        if score > 0:
            results.append((docs[idx], score))
            
    # Sort by score desc
    results.sort(key=lambda x: x[1], reverse=True)
    return results

if __name__ == "__main__":
    docs, vectorizer, tfidf_matrix = load_data()
    
    while True:
        query = input("\nEnter your search query (or empty to quit): ").strip()
        if not query:
            break

        results = search_solutions(query, docs, vectorizer, tfidf_matrix)

        print(f"\nTop {len(results)} results for: {query!r}\n")

        for doc, score in results:
            snippet = doc[TEXT_FIELD][:200] + ("…" if len(doc[TEXT_FIELD]) > 200 else "")
            print(f"- id={doc.get(ID_FIELD, '?')} | score={score:.4f}")
            print(f"  {snippet}\n")
