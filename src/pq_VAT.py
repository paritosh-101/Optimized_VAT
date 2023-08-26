import heapq
import networkx as nx
import numpy as np

def optimized_VAT_with_pq(R):
    N, _ = R.shape
    
    # Create a complete graph with dissimilarities as weights
    G = nx.complete_graph(N)
    for i in range(N):
        for j in range(i+1, N):
            G[i][j]['weight'] = R[i, j]

    # Start with the object pair with the maximum dissimilarity
    max_weight_edge = max(G.edges(data=True), key=lambda x: x[2]['weight'])
    I = [max_weight_edge[0]]

    # Use a priority queue to keep track of minimum dissimilarity for each remaining object
    pq = [(R[i, I[0]], i) for i in range(N) if i != I[0]]
    heapq.heapify(pq)

    while pq:
        dist, node = heapq.heappop(pq)
        I.append(node)

        # Update the priority queue with the dissimilarities of the remaining objects to the newly added object
        for idx, (d, n) in enumerate(pq):
            new_dist = R[n, node]
            if new_dist < d:
                pq[idx] = (new_dist, n)
        heapq.heapify(pq)

    # Reorder the matrix based on the obtained indices
    RI = np.argsort(I)
    RV = R[:, I][I, :]
    
    return RV, I, RI