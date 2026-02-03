import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import connected_components


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """
    
    ##MY ADDED ASSERTIONS

    #mst not fully connected
    n_components, _ = connected_components(mst, directed=False)
    assert n_components == 1, f'MST is disconnected into {n_components} components'

    #not symmetric
    #doing allclose instead of all because of precision issues with floats
    assert np.allclose(mst, mst.T), 'Proposed MST adjacency matrix is not symmetric'

    #num edges not valid (not n_vertices - 1)
    num_vertices = len(adj_mat)
    num_edges = len(mst.nonzero()[1])/2 #count the # nonzero elements (edge weights), then divide by 2 bc it should be symmetric
    assert num_edges == num_vertices-1, f'Proposed MST does not have the correct number of edges {num_vertices-1}'


    #ASSIGNMENT ASSERTION
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), f'Proposed MST has incorrect expected weight: was expecting {expected_weight}, got {total}'

    


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    Unit test for the construction of a minimum spanning tree on a graph that is disconnected (has two separate subgraphs).
    
    """
    file_path = './data/disconnected.csv'
    g = Graph(file_path)
    g.construct_mst()
    with pytest.raises(ValueError, match="disconnected"):
        check_mst(g.adj_mat, g.mst, 1)
