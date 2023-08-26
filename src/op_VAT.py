import numpy as np

def optimized_VAT(R):
    N, _ = R.shape
    I = [np.unravel_index(np.argmax(R), R.shape)[0]]  # get the row index of the maximum value in R
    J = list(range(N))
    J.remove(I[0])

    C = np.zeros(N, dtype=int)
    C[0] = 1
    cut = np.zeros(N)
    cut[0] = np.max(R)

    for r in range(1, N):
        Y = R[I, :][:, J]
        min_Y = np.min(Y, axis=0)
        j = np.argmin(min_Y)
        I.append(J[j])
        if r < N-1:
            C[r] = np.argmin(Y[:, j])
            cut[r] = min_Y[j]
        J.pop(j)

    RI = np.argsort(I)
    RV = R[:, I][I, :]
    
    return RV, C, I, RI, cut