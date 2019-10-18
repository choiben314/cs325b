import numpy as np

def curvature(points):
    radians = 0
    
    for i in range(1, len(points) - 1):
        A, B, C = points[i-1:i+2]
        a_magnitude = distance(B, C).m
        a_euclidean = np.subtract(B, C)
        a_euclidean *= np.divide(a_magnitude, np.linalg.norm(a_euclidean))
        c_magnitude = distance(A, B).m
        c_euclidean = np.subtract(A, B)
        c_euclidean *= np.divide(c_magnitude, np.linalg.norm(c_euclidean))
        cosine = np.dot(a_euclidean, c_euclidean) / (a_magnitude * c_magnitude)
        angle = np.arccos(cosine)
        radians += angle
    
    return radians