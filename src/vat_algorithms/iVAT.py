# import numpy as np
# from VAT_library.VAT import VAT
# # from VAT_library.pq_VAT import optimized_VAT_with_pq as VAT

# def iVAT(R, VATflag=False):
#     """
#     Input parameters:
#     R (n*n double): dissimilarity data input
#     VATflag (boolean): TRUE - R is VAT-reordered
    
#     Output Values:
#     RV (n*n double): VAT-reordered dissimilarity data
#     RiV (n*n double): iVAT-transformed dissimilarity data
#     """
#     N = R.shape[0]
#     reordering_mat = np.zeros(N, dtype=int)
#     reordering_mat[0] = 1
#     if VATflag:
#         RV = R
#         RiV = np.zeros((N, N))
#         for r in range(1, N):
#             c = np.arange(r)
#             y, i = np.min(RV[r, 0:r]), np.argmin(RV[r, 0:r])
#             reordering_mat[r] = i
#             RiV[r, c] = y
#             cnei = c[c != i]
#             RiV[r, cnei] = np.maximum(RiV[r, cnei], RiV[i, cnei])
#             RiV[c, r] = RiV[r, c]
#     else:
#         RV, C, _,_,_ = VAT(R)
#         # RV, C, _ = VAT(R)
#         RiV = np.zeros((N, N))
#         for r in range(1, N):
#             c = np.arange(r)
#             reordering_mat[r] = C[r]
#             RiV[r, c] = RV[r, C[r]]
#             cnei = c[c != C[r]]
#             RiV[r, cnei] = np.maximum(RiV[r, cnei], RiV[C[r], cnei])
#             RiV[c, r] = RiV[r, c]
#     return RiV, RV, reordering_mat


import numpy as np
from tqdm import tqdm
# from VAT import VAT
# from VAT_library.pq_VAT import optimized_VAT_with_pq as VAT

def VAT(R):
  
    N, M = R.shape
    K = np.arange(N)
    J = K
    # P=zeros(1,N)
    y, i = np.max(R, axis=0), np.argmax(R, axis=0)
    y, j = np.max(y), np.argmax(y)
    I = i[j]
    J = np.delete(J, I)
    
    y, j = np.min(R[I, J]), np.argmin(R[I, J])
    I = np.append(I, J[j])
    J = np.delete(J, j)
    C = np.zeros(N, dtype=int)
    C[0:2] = 1
    cut = np.zeros(N)
    cut[1] = y
    for r in tqdm(range(2, N-1), desc="VAT Processing"):
    # for r in range(2, N-1):
        # y, i = np.min(R[I, J], axis=0), np.argmin(R[I, J], axis=0)
        y = np.zeros(len(J))
        i = np.zeros(len(J))
        for k in tqdm(range(len(J)), desc="Inner loop", leave=False):
        # for k in range(len(J)):
            y[k], i[k] = np.min(R[I, J[k]]), np.argmin(R[I, J[k]])

        y, j = np.min(y), np.argmin(y)
        I = np.append(I, J[j])
        J = np.delete(J, j)
        C[r] = i[j]
        cut[r] = y
    y, i = np.min(R[I, J], axis=0), np.argmin(R[I, J], axis=0)
    I = np.append(I, J)
    C[N-1] = i
    cut[N-1] = y
    RI = np.zeros(N, dtype=int)
    for r in tqdm(range(N), desc="Final loop"):
    # for r in range(N):
        RI[I[r]] = r
    RV = R[I, :]
    RV = RV[:, I]
    return RV, C, I, RI, cut

def iVAT(R, VATflag=False):
    """
    Input parameters:
    R (n*n double): dissimilarity data input
    VATflag (boolean): TRUE - R is VAT-reordered
    
    Output Values:
    RV (n*n double): VAT-reordered dissimilarity data
    RiV (n*n double): iVAT-transformed dissimilarity data
    """
    N = R.shape[0]
    reordering_mat = np.zeros(N, dtype=int)
    reordering_mat[0] = 1
    if VATflag:
        RV = R
        RiV = np.zeros((N, N))
        for r in tqdm(range(1, N), desc="Processing"):
        # for r in range(1, N):
            c = np.arange(r)
            y, i = np.min(RV[r, 0:r]), np.argmin(RV[r, 0:r])
            reordering_mat[r] = i
            RiV[r, c] = y
            cnei = c[c != i]
            RiV[r, cnei] = np.maximum(RiV[r, cnei], RiV[i, cnei])
            RiV[c, r] = RiV[r, c]
    else:
        RV, C, _,_,_ = VAT(R)
        # RV, C, _ = VAT(R)
        RiV = np.zeros((N, N))
        for r in tqdm(range(1, N), desc="iVAT Processing"):
        # for r in range(1, N):
            c = np.arange(r)
            reordering_mat[r] = C[r]
            RiV[r, c] = RV[r, C[r]]
            cnei = c[c != C[r]]
            RiV[r, cnei] = np.maximum(RiV[r, cnei], RiV[C[r], cnei])
            RiV[c, r] = RiV[r, c]
    return RiV, RV, reordering_mat