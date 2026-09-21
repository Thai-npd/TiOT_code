import numpy as np
import ot
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, eye, kron
np.seterr(divide='ignore', invalid='ignore', over='ignore')
import time
import warnings
from scipy.stats import norm

def normalization(x,y):
    """
    Normalize input time series by formular
    x = (x - mean(x)) / std(x)

    Parameters
        x (ndarray): 1D array of shape (n,)
        y (ndarray): 1D array of shape (m,)

    Returns:
        (tuple): normalized x, normalized y
    """

    return (x - np.mean(x)) / np.std(x), (y - np.mean(y)) / np.std(y)
 

def costmatrix0(x ,y, w) -> np.ndarray : 
    """
    Cost matrix define in the TAOT's way: C.

    Parameters:
        x (ndarray): 1D array of shape (n,)
        y (ndarray): 1D array of shape (m,)
        w (float)  : parameter to calculate C_ij
 
    Returns:
        ndarray: 2D array of shape (n,m)
    """

    spatial_cost, temporal_cost = spat_temp_cost(x,y)
    C = spatial_cost + w * temporal_cost
    C = C / np.median(C)
    return C  

def costmatrix1(x,y, w) -> np.ndarray :
    """
    Cost matrix define in the TiOT's way: C.

    Parameters:
        x (ndarray): 1D array of shape (n,)
        y (ndarray): 1D array of shape (m,)
        w (float)  : parameter to calculate C_ij

    Returns:
        ndarray: 2D array of shape (n,m)
    """
    x,y = normalization(x,y)
    spatial_cost, temporal_cost = spat_temp_cost(x,y)
    return w * spatial_cost + (1-w)*temporal_cost

def costmatrix2(x,y, w) -> np.ndarray :
    """
    Cost matrix define in the TiOT's way, followed by the TAOT-style median
    normalization: C = C / median(C). The normalization multiplies the effective
    entropic eps by median(C) (K = exp(-C/eps)), which speeds up Sinkhorn at small
    eps and puts every pair's distances on that pair's own scale.

    Parameters:
        x (ndarray): 1D array of shape (n,)
        y (ndarray): 1D array of shape (m,)
        w (float)  : parameter to calculate C_ij

    Returns:
        ndarray: 2D array of shape (n,m)
    """
    C = costmatrix1(x, y, w)
    return C / np.median(C)

def spat_temp_cost(x,y, eps = 1):
    n, m = len(x), len(y)
    t,s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    if x.ndim == 1 and y.ndim == 1:
        spatial_cost =  (x[:, None] - y[None, :])**2 / eps
    else:
        x_norm2 = np.sum(x**2, axis=1)[:, None]
        y_norm2 = np.sum(y**2, axis=1)[None, :]
        spatial_cost = (x_norm2 + y_norm2 - 2 * x @ y.T) / eps
    temporal_cost =  (t[:, None] - s[None, :])**2 / eps
    return spatial_cost, temporal_cost
 
 
def TiOT(x, y, a=None, b=None, detail_mode=False, verbose=False, timing=False):
    """
    Solve the Time-integrated Optimal Transport (TiOT) problem between two discrete distributions.
    Uses sparse matrices for better scalability.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        detail_mode (bool, optional): If False (default), returns distance, optimal w.
                                      If True, returns distance, transport plan, optimal w.
        timing (bool, optional): If True, record elapsed time and return it.

    Returns:
        tuple: 
            - If `detail_mode=False`: returns distance and optimal w.
            - If `detail_mode=True`: returns distance, transport plan, optimal w.
            - If `timing=True`: additionally return elapsed time.
    """

    start = time.perf_counter()
    n, m = len(x), len(y)

    # Default uniform weights if not provided
    if a is None:
        a = np.ones(n) / n
    if b is None:
        b = np.ones(m) / m

    # Normalize time indices (dummy normalization function assumed available)
    t, s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    x, y = normalization(x, y)

    # Sparse constraint blocks
    A1 = -kron(eye(n, format="csr"), np.ones((m,1), dtype=np.float32))   # - kron(I_n, 1_m)
    A2 = -kron(np.ones((n,1), dtype=np.float32), eye(m, format="csr"))   # - kron(1_n, I_m)

    # Spatial and temporal costs (assumed provided)
    spatial_cost, temporal_cost = spat_temp_cost(x, y)

    # Extra column (sparse version)
    extraCol = (temporal_cost - spatial_cost).reshape(-1, 1)
    extraCol = csr_matrix(extraCol)

    # Combine everything into one sparse constraint matrix
    A = hstack([A1, A2, extraCol], format="csr")
    sparse_mem = A.data.nbytes + A.indptr.nbytes + A.indices.nbytes
    # Cost vector
    c = np.concatenate((a, b, [0]))

    # Bounds: (None, None) for the transport vars, (0,1) for w
    bounds = [(None, None)] * (n + m) + [(0, 1)]

    # Solve with HiGHS
    res = linprog(c, A_ub=A, b_ub=temporal_cost.flatten(), bounds=bounds, method="highs")

    end = time.perf_counter()

    # Handle outputs
    if timing:
        if not detail_mode:
            return -res.fun, res.x[-1], end - start
        else:
            return -res.fun, TAOT(x, y, w=res.x[-1])[1], res.x[-1], end - start
    else:
        if not detail_mode:
            return -res.fun, res.x[-1]
        else:
            return -res.fun, TAOT(x, y, w=res.x[-1])[1], res.x[-1]       


def eTiOT(x, y, a = None, b = None, eps = 0.01, maxIter = 5000, tolerance = 0.005, solver = 'PGD', eta = 10**-2, init_stepsize = True, submax_iter = 50,  subprob_tol = 10**-7, freq = 20, verbose = False, timing = False):
    """
    Solves the entropic Time-integrated Optimal Transport (eTiOT) problem use block coordinate descent.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        eps (float, optional): Entropic regularization parameter. Default is 0.01.
        maxIter (int, optional): Maximum number of iterations for BCD. Default is 5000.
        tolerance (float, optional): Convergence tolerance for BCD. Default is 0.005.
        solver (str, optional): Method used for finding optimal w. Default is 'newton'.
        subprob_tol (float, optional): Tolerance for solving the inner subproblem (e.g., newton). Default is 1e-7.
        freq (int, optional): Frequency (in iterations) at which the weight variable `w` is updated. Default is 1.
        verbose (bool, optional): If True, prints progress. Default is False.
        timing (bool, optinal): If True, record the elapsed time and return it

    Returns:
        tuple: distance, transport plan, optimal w
    """
    start = time.perf_counter()
    n, m = len(x), len(y)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    x,y = normalization(x,y)
    spatial_cost, temporal_cost = spat_temp_cost(x,y,eps)
    spat_temp_diff = temporal_cost - spatial_cost
    
    def PGD(g,h, w, subprob_tol = 10**-7, maxIter = 50, eta = 10**-2, init_stepsize = False):
        def f(w):
            C = w*spatial_cost + (1-w)*temporal_cost
            K = np.exp(-C/eps)
            return  -eps * np.log(g) @ a - eps * np.log(h) @ b + eps * (g.T @ K @ h)
        def df(w):
            # The formulation for the derivative is df(w) = g.T @ (spat_temp_diff * K_w) @ h. To optimize the efficiency the variable temp(temporary) is used through out the entire process.
            temp = w * spat_temp_diff
            temp -= temporal_cost
            np.exp(temp, out= temp)
            temp *= spat_temp_diff
            return eps *( g.T @ temp @ h)
        
        def df2w(w):
            temp = w * spat_temp_diff
            temp -= temporal_cost
            np.exp(temp, out= temp)
            temp *= spat_temp_diff**2
            return  eps* (g.T @ (temp @ h))
        
        def proj(w):
            if w > 1:
                w = 1
            elif w < 0:
                w = 0
            return w 
        def init_eta(df2w):
            possible_stepsize = 1/df2w
            if possible_stepsize >= 10:
                return possible_stepsize/20
            else:
                return possible_stepsize/10
        if init_stepsize:
            eta = init_eta(df2w(w))

        for i in range(maxIter):
            w_prev = w
            dfw = df(w)
            w = proj(w - eta*dfw)
            if np.abs(w-w_prev) < subprob_tol:
                break
        K = w * spat_temp_diff
        K -= temporal_cost
        np.exp(K, out = K) 
        # The three lines above is to compute K efficiently with the first two lines computing the cost negative C and the third line compute exp(nC).
        if i == maxIter - 1 and verbose >= 1 : print(f"PGD Algorithm does not converge after {i} iterations with stepsize {eta} and dfw = {dfw}") # and verbose >= 1
        return w, K

    g, h, w = np.ones(n)/n , np.ones(m)/m , 0.5
    nC = w * spat_temp_diff - temporal_cost
    K = np.exp(nC)
    curIter = 0
    diverged = False
    while curIter < maxIter:
        g = a/ (K @ h)
        h = b/(K.T @ g)
        # Stop as soon as an iterate becomes NaN. At small eps, K = exp(-C/eps) underflows
        # to 0, so g = a/(K@h) blows up to inf and the next 0*inf produces NaN. Once any
        # entry is NaN the BCD is an absorbing NaN state -- no operation in the loop can
        # bring it back to finite (proj(NaN)=NaN, exp(NaN)=NaN) -- so continuing to maxIter
        # only wastes time. Bail out and flag the solve as failed (distance = inf). This
        # check is a no-op for finite runs, so converging pairs are unaffected.
        if np.any(np.isnan(g)) or np.any(np.isnan(h)):
            diverged = True
            if verbose >= 1:
                warnings.warn(f"eTiOT diverged (NaN iterate) at iteration {curIter}: "
                              f"eps={eps} is too small for this pair; returning inf.")
            break
        if curIter % freq ==0 :
            w, K = PGD(g,h, w, subprob_tol= subprob_tol, maxIter=submax_iter, eta=eta, init_stepsize=init_stepsize)
        if curIter % freq == 0 and np.sum(np.abs(g * (K @ h) - a)) < tolerance:
            if verbose >= 1:
                print(f"TiOT-BCD Algorithm converges after {curIter+1} iterations ")
            break
        curIter += 1
    if curIter == maxIter and verbose >= 1: print(f"TiOT algorithm did not stop after {maxIter} iterations")
    if diverged:
        distance = np.inf
    else:
        C = eps * (temporal_cost - w*spat_temp_diff)
        K *= h.reshape((1, -1))
        K *= g.reshape((-1,1))
        #transport_plan = g.reshape((-1, 1)) * K * h.reshape((1, -1))
        distance = np.einsum("ij,ij->", C, K)
    end = time.perf_counter()
    if timing == True:
        return distance, K, w, end - start
    else:
        return distance, K, w


def eTAOT2(x, y, a = None, b = None, w = 0.5, eps = 0.01, maxIter=5000, tolerance=0.005, freq = 20, verbose = False, timing = False):
    """
    eTAOT with the median-normalized TiOT cost matrix (costmatrix2): identical to
    eTAOT(..., costmatrix=costmatrix2). Kept as a named variant so eTAOT (raw
    costmatrix1 scale) and eTAOT2 (median-normalized) can be compared directly.

    Parameters / returns: same as eTAOT.
    """
    return eTAOT(x, y, a=a, b=b, w=w, eps=eps, costmatrix=costmatrix2,
                 maxIter=maxIter, tolerance=tolerance, freq=freq,
                 verbose=verbose, timing=timing)


def TAOT(x, y, a = None, b = None, w = 0.5, costmatrix = costmatrix1, verbose = None, timing = False):
    """
    Solve the Time Adaptive Optimal Transport (TAOT) problem between two empirical distributions.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        w (float, optional): Weight parameter to compute cost matrix C.
        costmatrix (callable, optional): A function to compute the cost matrix between x and y. Default is `costmatrix1`.
        timing (bool, optinal): If True, record the elapsed time and return it

    Returns:
        tuple: distance, transport_plan
    """
    start = time.perf_counter()
    n, m = len(x), len(y)
    M = costmatrix(x, y, w)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    result = ot.lp.emd2(a,b,M, numItermax = 10**8, return_matrix=True) 
    distance = result[0]
    transport_plan = result[1]['G']
    end = time.perf_counter()
    if timing == True:
        return distance, transport_plan, end -start
    else:
        return distance, transport_plan


def eTAOT(x, y, a = None, b = None, w = 0.5, eps = 0.01, costmatrix = costmatrix1,  maxIter=5000, tolerance=0.005, freq = 20, verbose = False, timing = False):
    """
    Solves the entropic Time Adaptive Optimal Transport (eTAOT) problem between two empirical distributions.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        w (float, optional): Weight parameter to compute cost matrix C.
        eps (float, optional): Entropic regularization parameter. Default is 0.01.
        costmatrix (callable, optional): Function to compute the cost matrix between `x` and `y`. Default is `costmatrix1`.
        maxIter (int, optional): Maximum number of iterations for BCD. Default is 5000.
        tolerance (float, optional): Convergence tolerance for BCD. Default is 0.005.
        verbose (bool, optional): If True, prints progress and debug information. Default is False.
        timing (bool, optinal): If True, record the elapsed time and return it
    Returns:
        tuple: distance, transport_plan
    """
    start = time.perf_counter()
    n, m = len(x), len(y)
    C = costmatrix(x, y, w)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    K = np.exp(-C/eps)
    g = np.ones(n) / n
    h = np.ones(m) / m
    curIter = 0
    while curIter < maxIter:
        g = a/ (K @ h)
        h = b/(K.T @ g)
        if  curIter % freq == 0 and np.sum(np.abs(g * (K @ h) - a)) < tolerance:
            if verbose >= 1: 
                print(f"TAOT-BCD Algorithm converges after {curIter+1} iterations")
            break
        curIter += 1
    if verbose >= 1:
        if curIter == maxIter: print(f"TAOT algorithm did not stop after {maxIter} iterations")
    transport_plan = g.reshape((-1, 1)) * K * h.reshape((1, -1))
    distance = np.sum(C * transport_plan)
    distance = np.einsum("ij,ij->", C, transport_plan)
    end = time.perf_counter()
    if timing == True:
        return distance, transport_plan, end - start
    else:
        return distance, transport_plan


