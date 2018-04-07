import numpy as np

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    distance = 0.0
    
    # Compute the dot product between u and v
    dot = np.dot(u.T,v)
    # Compute the L2 norm of u
    norm_u = np.sqrt(np.sum(np.square(u)))
    
    # Compute the L2 norm of v
    norm_v = np.sqrt(np.sum(np.square(v)))
    # Compute the cosine similarity
    cosine_similarity = dot / (norm_u*norm_v)
    #print(dot.shape,norm_u.shape,norm_v.shape)
    
    return cosine_similarity
