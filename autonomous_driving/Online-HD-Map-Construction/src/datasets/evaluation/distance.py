from scipy.spatial import distance
from numpy.typing import NDArray

def chamfer_distance(line1: NDArray, line2: NDArray) -> float:
    ''' Calculate chamfer distance between two lines. Make sure the 
    lines are interpolated.

    Args:
        line1 (array): coordinates of line1
        line2 (array): coordinates of line2
    
    Returns:
        distance (float): chamfer distance
    '''
    
    dist_matrix = distance.cdist(line1, line2, 'euclidean')
    dist12 = dist_matrix.min(-1).sum() / len(line1)
    dist21 = dist_matrix.min(-2).sum() / len(line2)

    return (dist12 + dist21) / 2

def frechet_distance(line1: NDArray, line2: NDArray) -> float:
    ''' Calculate frechet distance between two lines. Make sure the 
    lines are interpolated.

    Args:
        line1 (array): coordinates of line1
        line2 (array): coordinates of line2
    
    Returns:
        distance (float): frechet distance
    '''
    
    raise NotImplementedError

