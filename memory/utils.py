import numpy as np
from typing import List

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    :param vec_a: List of floats representing the first vector.
    :param vec_b: List of floats representing the second vector.
    :return: Cosine similarity score.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

def sort_items_by_relevance(query_embeddings: List[float], items_embeddings: List[List[float]]) -> [List[float], List[int]]:
    """
    Find the most relevant items to a given query based on cosine similarity of embeddings.
    :param query_embeddings: The keyword you are querying for as an embedding.
    :param items_embeddings: A list of embeddings of items to compare against.
    :return: Similarities and the Indices of items sorted by relevance (highest relevance first).
    """
    # Convert lists to numpy arrays for efficient computation
    query_vector = np.array(query_embeddings)
    items_matrix = np.array(items_embeddings)
    
    # Normalize vectors to unit length to compute cosine similarity
    query_norm = np.linalg.norm(query_vector)
    items_norm = np.linalg.norm(items_matrix, axis=1)
    
    # Handle potential division by zero for poorly formed embeddings
    query_vector = query_vector / query_norm if query_norm != 0 else query_vector
    items_matrix = items_matrix / items_norm[:, None]  # Normalize each row
    
    # Compute cosine similarity (dot product of normalized vectors)
    similarities = items_matrix @ query_vector
    
    # Sort indices by similarity in descending order
    sorted_indices = np.argsort(similarities)[::-1]
    return similarities, sorted_indices

if __name__ == "__main__":
    # Example embeddings
    query = [0.1, 0.2, 0.3]
    items = [
        [0.2, 0.1, 0.4],
        [0.9, 0.7, 0.2],
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    for item in items:
        print(cosine_similarity(query, item))

    similarities, sorted_indices = sort_items_by_relevance(query, items)
    print(similarities, sorted_indices)