import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        num_vertices =len(self.adj_mat)

        #initializing
        in_mst = [False] * num_vertices
        weights = [float("inf")] * num_vertices     #min weight edge to reach each vertex
        predecessors = [None] * num_vertices
        
        
        s_idx = 0  #starting from the 0th vertex
        weights[s_idx] = 0  #the cost to reach the first index is 0 :)
        
        final_mst = np.zeros((num_vertices, num_vertices))      #initialize adj matrix to all 0's

        pq = [(weights[v], v) for v in range(num_vertices)]     #make priority queue (weight then vertex bc it sorts by first element)
        heapq.heapify(pq)

        # While priority queue has vertices
        while pq:
            u_weight, u = heapq.heappop(pq)  # Pop the vertex with minimum weight

            #if it's in the mst, skip it
            if in_mst[u]:
                continue
            
            in_mst[u] = True    # add vertex u to MST
            
            # add edge to mst (if not the starting vertex)
            if predecessors[u] is not None:
                pred = predecessors[u]
                final_mst[u][pred] = self.adj_mat[u][pred]
                final_mst[pred][u] = self.adj_mat[pred][u]  #it's symmetric so add both ways
            
            # update neighbors of u
            for v in range(num_vertices):
                edge_weight = self.adj_mat[u][v]
                
                # if edge exists AND v is not in mst AND this edge is better
                if edge_weight > 0 and not in_mst[v] and edge_weight < weights[v]:
                    weights[v] = edge_weight
                    predecessors[v] = u
                    heapq.heappush(pq, (weights[v], v))

        # update the actual mst!!
        self.mst = final_mst
