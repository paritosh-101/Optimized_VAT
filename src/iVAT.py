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
from VAT_library.VAT import VAT
# from VAT_library.pq_VAT import optimized_VAT_with_pq as VAT

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