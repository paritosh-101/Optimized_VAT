import numpy as np
import heapq
import networkx as nx

def fast_VAT(R):
    N, M = R.shape
    
    # Create a graph from the dissimilarity matrix
    G = nx.Graph()
    for i in range(N):
        for j in range(i+1, N):  # We only need upper triangle of matrix
            G.add_edge(i, j, weight=R[i, j])
    
    # Compute the minimum spanning tree
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    
    # Extract the edges and weights from MST
    mst_edges = list(mst.edges(data=True))
    mst_edges_sorted = sorted(mst_edges, key=lambda x: x[2]['weight'])
    
    # Sort nodes based on their MST edge weights
    I = [mst_edges_sorted[0][0], mst_edges_sorted[0][1]]
    C = [1]
    cut = [mst_edges_sorted[0][2]['weight']]
    
    for edge_data in mst_edges_sorted[1:]:
        node1, node2, data = edge_data
        weight = data['weight']
        
        if node1 not in I:
            I.append(node1)
            C.append(node2)
        else:
            I.append(node2)
            C.append(node1)
        cut.append(weight)
    
    # Convert I to a permutation index
    RI = np.argsort(I)
    RV = R[np.ix_(I, I)]
    
    return RV, C, I, RI, cut

# Testing the optimized VAT function
R = np.array([[0, 2, 3, 4], [2, 0, 5, 6], [3, 5, 0, 7], [4, 6, 7, 0]])
RV, C, I, RI, cut = fast_VAT(R)
print("RV:\n", RV)
print("C:\n", C)
print("I:\n", I)
print("RI:\n", RI)
print("cut:\n", cut)