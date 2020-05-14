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

def interpolate_sinc(xi, yi, zi, a=2, s=1, h=0.01):
    """
    Uses sinc (lanczos) interpolation to interpolate the function f(x_i) = y_i onto the points z_i, i.e. returns
    f(z_i)

    Inputs:
    - xi    : Mx3
    - yi    : MxK
    - zi    : Nx3

    Outputs:
    - f(zi) : NxK
    """

    def lanczos(x):
        """
        Lanczos window function. x: _xN
        """
        bound = a*s
        mask = np.abs(x) <= bound
        x_t = np.sinc(x/s) * np.sinc(x/(a*s))
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
    # least squares to get coefficients for sinc  (ridge least squares)

    lhs = sinc.T@sinc + h*np.identity(sinc.shape[1])
    rhs = sinc.T@yi
    c = np.linalg.solve(lhs, rhs)

    # get f(zi) using the coefficients
    Lzi = generate_sinc_matrix(zi, zi)

    return Lzi@c
    

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
    Sq_cc = interpolate_sinc(qvecs, Sq, grid_samples, a=4, s=side_samples[1]-side_samples[0], h=0.3)

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

