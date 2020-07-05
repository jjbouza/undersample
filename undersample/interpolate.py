"""
Description: Various interpolation algorithms for non-uniform data, mostly for use in dMRI q-space interpolation.
Author: Jose J. Bouza
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.sparse.linalg import lsqr
from scipy.spatial import ConvexHull


def get_convex_cube(vecs):
    """
    Returns the side length of an insctribed cube in the convex hull of vecs.
    """
    # figure out the largest grid that fits inside the convex hull of the q-space samples (approximately) by inscribing a cube inside
    # an inscribed sphere

    # start by getting the convex hull of the qvecs
    convex_hull = ConvexHull(vecs)

    # find the radius of the smallest norm vertex of the convex hull
    convex_hull_vertex_norms = np.linalg.norm(vecs[convex_hull.vertices], axis=1)
    smallest_vertex_norm = np.min(convex_hull_vertex_norms)

    # insribe a cube in the sphere of radius smallest_vertex_norm
    cube_side_length = 2*smallest_vertex_norm/np.sqrt(3) 

    return cube_side_length

def cost(x, A, b, w):
    return np.linalg.norm(A@x-b)+np.linalg.norm(w*x)

def cost_prime(x, A, y, w):
    t_0 = ((A).dot(x) - y)
    t_1 = (w * x)
    t_2 = np.linalg.norm(t_1)
    gradient = ((2 * (A.T).dot(t_0)) + ((1 / t_2) * (t_1 * w)))
    return gradient

def interpolate_sinc(xi, yi, zi, a=1):
    """
    Uses sinc (lanczos) interpolation to interpolate the function f(x_i) = y_i onto the points z_i, i.e. returns
    f(z_i)

    
    * THIS IS MY CUSTOM IMPLEMENTATION BASED ON MY UNDERSTANDING OF THE PAPER.
    Inputs:
    - xi    : Mx3
    - yi    : MxK
    - zi    : Nx3

    Outputs:
    - f(zi) : NxK
    """
    from scipy.optimize import fmin
    from scipy.optimize import nnls
    import multiprocessing
    import os

    def lanczos(x, bound=np.inf):
        """
        Lanczos window function. x: _xN
        """
        mask = np.abs(x) <= bound
        x_t = np.sinc(x/a)
        lanczos = x_t*mask
        lanczos = np.prod(lanczos, axis=-1)

        return lanczos

    def generate_sinc_matrix(xi, ki):
        X_matrix = np.stack([xi for i in range(ki.shape[0])], axis=1)
        K_matrix = np.stack([ki for i in range(X_matrix.shape[0])], axis=0)
        supp = X_matrix-K_matrix
        sinc = lanczos(supp)
        return sinc

    sinc = generate_sinc_matrix(xi, zi)
    x0 = np.zeros([yi.shape[1], zi.shape[0]])
    for i in range(yi.shape[1]):
        x0[i] = nnls(sinc, yi[:,i], maxiter=1e10)[0]

    w = np.sum(zi**2, axis=-1)
    p = multiprocessing.Pool(os.cpu_count())
    out = p.starmap(fmin, [(cost, x0[i], (sinc, yi[:,i],w), 0.0001, 0.0001, 1000, None, False, True) for i in range(yi.shape[1])]) 
    p.close()
    p.join()
    
    fzi = np.stack(out)

    return fzi
    

def interpolate_sinc_fmin(xi, yi, zi, h=2):
    """
    Uses sinc (lanczos) interpolation to interpolate the function f(x_i) = y_i onto the points z_i, i.e. returns
    f(z_i)

    * THIS IS A DIRECT COPY OF WENXINGS CODE. I AM UNSURE ABOUT SOME PARTS, FOR EXAMPLE THE NP.KRON CALL.

    Inputs:
    - xi    : Mx3
    - yi    : MxK
    - zi    : Nx3

    Outputs:
    - f(zi) : NxK
    """
    from scipy.optimize import fmin_cg
    from scipy.optimize import nnls
    import multiprocessing
    import os
    
    temp = np.ones([1, zi.shape[0], 1])
    A = np.zeros([xi.shape[0], zi.shape[0]])

    pdata = np.kron(xi[:,None,:], temp)

    matrix = zi[None].repeat(pdata.shape[0], 0)-pdata/h
    sinc = np.sinc(matrix)
    sinc_vals = np.product(sinc, axis=-1)

    x0 = np.zeros([yi.shape[1], zi.shape[0]])
    for i in range(yi.shape[1]):
        x0[i] = nnls(sinc_vals, yi[:,i])[0]

    w = np.sum(zi**2, axis=-1)
    p = multiprocessing.Pool(os.cpu_count())
    out = p.starmap(fmin_cg, [(cost, x0[i], cost_prime, (sinc_vals, yi[:,i],w),0.0001,np.inf,0.00001,1000) for i in range(yi.shape[1])])
    p.close()
    p.join()

    fzi = np.stack(out)

    return fzi

def interpolate_q_space_sinc(Sq, qvecs, nsamples):
    """
    Performs sinc (well, really Lanzcos) interpolation of arbitrarly sampled q-space data. Rather than use an iterative algorithm we directly
    solve the linear system S(x) = \sum_k c_k sinc(x-k) since it is usually small enough.

    Inputs:
    - Sq = S(k,q) : _xM
    - qvecs : Mx3

    Outputs:
    - Sq_cc : _xN
    """
    Sq_s = Sq.shape

    Sq = Sq.reshape(-1, Sq.shape[-1]).T

    cube_side_length = get_convex_cube(qvecs)

    # generate 3d grid in [-cube_side_length, cube_side_length]^3 with nsamples samples per dimension
    side_samples = np.linspace(-cube_side_length/2+cube_side_length/100, cube_side_length/2-cube_side_length/100, num=nsamples, endpoint=True)

    # generate grid samples
    grid_samples = np.array([(x,y,z) for x in side_samples for y in side_samples for z in side_samples])
    Sq_cc = interpolate_sinc(qvecs, Sq, grid_samples, a=side_samples[1]-side_samples[0])

    return Sq_cc.T.reshape(*Sq_s[:-1], -1), grid_samples

def interpolate_q_space_linear(Sq, qvecs, nsamples):
    """
    Performs linear interpolation of arbitrarly sampled q-space data, for example from a multi-shell aquisition. We insribe a cube inside the 
    convex hull of the q-vectors and generate a regular grid in this cube. We then interpolate Skq on this regular grid.
    """
    Sq_s = Sq.shape
    Sq = Sq.reshape(-1, Sq.shape[-1]).T

    cube_side_length = get_convex_cube(qvecs)

    # generate 3d grid in [-cube_side_length, cube_side_length]^3 with nsamples samples per dimension
    side_samples = np.linspace(-cube_side_length/2+cube_side_length/100, cube_side_length/2, num=nsamples, endpoint=False)

    # generate grid samples
    grid_samples = np.array([(x,y,z) for x in side_samples for y in side_samples for z in side_samples])

    # perform linear interpolation
    interpolated_data = griddata(qvecs, Sq, grid_samples, method='linear')

    return interpolated_data.T.reshape(*Sq_s[:-1], -1), grid_samples

