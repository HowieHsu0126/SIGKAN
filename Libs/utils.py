from __future__ import absolute_import, unicode_literals, division, print_function
import numpy as np

def load_w2v_feature(file, max_idx=0):
    """
    Loads word2vec features from a file into a numpy array. Each line in the file represents
    a word vector, where the first line contains the number of vectors (n) and the vector 
    dimensions (d). Subsequent lines contain an index followed by the feature values.

    Args:
        file (str): Path to the word2vec feature file.
        max_idx (int, optional): The maximum index for feature allocation. Defaults to 0.

    Returns:
        np.ndarray: A 2D numpy array of shape (max(n, max_idx+1), d) with word vectors.
    """
    with open(file, "rb") as f:
        first_line = f.readline().strip().split()
        # Read the number of vectors (n) and vector dimension (d)
        n, d = int(first_line[0]), int(first_line[1])
        
        # Pre-allocate the feature matrix with zeros
        feature = np.zeros((max(n, max_idx + 1), d), dtype=np.float32)

        # Process the remaining lines to fill the feature matrix
        for line in f:
            content = line.strip().split()
            index = int(content[0])  # The index of the vector
            # Ensure feature matrix is large enough to include this index
            if index >= len(feature):
                feature = np.vstack([feature, np.zeros((index - len(feature) + 1, d))])

            # Assign values from the file to the corresponding row in the feature matrix
            feature[index, :] = np.array([float(x) for x in content[1:]], dtype=np.float32)

    # Ensure all vectors are of the same length (d)
    assert all(len(item) == d for item in feature), "All vectors must have the same dimension."
    
    return feature
