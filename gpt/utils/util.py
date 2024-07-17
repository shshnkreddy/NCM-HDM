import numpy as np 

def cosine_similarity(q1, q2):
        # Ensure that the arrays have the same length
    if len(q1) != len(q2):
        raise ValueError("Vectors must have the same length.")

    # Calculate the dot product
    dot_product = np.dot(q1, q2)

    # Calculate the magnitudes
    magnitude1 = np.linalg.norm(q1)
    magnitude2 = np.linalg.norm(q2)

    # Calculate the cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity

def minmax_normalize(array):
    # Calculate the minimum and maximum values
    min_val = np.min(array)
    max_val = np.max(array)

    # Avoid division by zero
    if min_val == max_val:
        return np.zeros_like(array)

    # Apply min-max normalization formula
    normalized_array = (array - min_val) / (max_val - min_val)

    return normalized_array

def decay(array, gamma):
    return array*np.power(gamma, np.arange(len(array)))

def convert_numpy_to_list(array):
    return array.tolist() if isinstance(array, np.ndarray) else array

