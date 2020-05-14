import os
import math

import numpy as np
import scipy.io as sio
from scipy.ndimage import map_coordinates

from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.core.gradients import gradient_table, GradientTable
from dipy.viz import window, actor
from dipy.reconst.dsi import pdf_odf
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel, DiffusionSpectrumModel
from dipy.data import get_sphere


import sys

def load_slice(fname, dname="data.nii.gz", xy_size=18, gtab=False, load=False):
    data, affine = load_nifti(os.path.join(fname, dname))
    bvals, bvecs = read_bvals_bvecs(os.path.join(fname, 'bvals'), os.path.join(fname, 'bvecs'))
    
    # generate q-vectors
    qvecs = []
    for bval, bvec in zip(bvals, bvecs):
        qvecs.append(bval*bvec)
    
    qvecs = np.stack(qvecs)
    if gtab:
        gt = gradient_table(bvals, bvecs)
        return data[data.shape[0]//2 +20 - xy_size:data.shape[0]//2 +20 +xy_size, data.shape[1]//2-xy_size:data.shape[1]//2+xy_size, data.shape[2]//2], qvecs, gt

    return data[data.shape[0]//2 - xy_size:data.shape[0]//2+xy_size, data.shape[1]//2-xy_size:data.shape[1]//2+xy_size, data.shape[2]//2], qvecs

def gtable_from_qvecs(qvecs, b0_threshold=10):
    """
    qvecs: Nx3
    """

    return GradientTable(qvecs, b0_threshold)

def vis_2d_field(odf, sphere):
    """
    Visualize a 2D ODF field.
    """

    r = window.Renderer()
    sfu = actor.odf_slicer(odf.reshape(1, *odf.shape), sphere=sphere, colormap='plasma', scale=0.5)
    sfu.display(x=0)
    r.add(sfu)
    window.show(r)


def save_mat(fname, dname, xy_size=None):
    """
    Load a diffusion image and save it in correct format for usage with jesses EAP estimator. 
    """
    data, affine = load_nifti(os.path.join(fname, dname))
    bvals, bvecs = read_bvals_bvecs(os.path.join(fname, 'bvals'), os.path.join(fname, 'bvecs'))

    if xy_size != None:
        data = data[data.shape[0]//2 - xy_size:data.shape[0]//2+xy_size, data.shape[1]//2-xy_size:data.shape[1]//2+xy_size, data.shape[2]//2]

    assert(len(data.shape) == 3 or len(data.shape) == 4)

    if len(data.shape) == 3:
        data = data[:, :, np.newaxis, :]        

    # [numberOfDirections, numberOfSlices, dimOfX, dimOfY]
    data = data.transpose(3,2,0,1)    
    bvals = bvals[:, np.newaxis]

    # save
    sio.savemat("signal.mat", {'Sig': data})
    sio.savemat("bdata.mat", {'bvals' : bvals, 'bvecs' : bvecs})

    print(data.shape)
    print("Done!")


# convert pdfs to odf and plot using dipy
def get_odf(signal, sphere):
    rstart = 0
    rend   = 5
    rstep  = 0.2
    radis = np.arange(rstart, rend, rstep)
    interp_coords =  radis*sphere.vertices[np.newaxis].T
    PrRadial = map_coordinates(signal, interp_coords, order=1)
    odf = (PrRadial * radis ** 2).sum(-1)

    return odf

def visualize_pdf(pdf, gtab=None):
    signal = pdf
    s = signal.shape
    grid_s = math.ceil((signal.shape[-1])**(1/3))
    signal = signal.reshape(*s[:-1], grid_s, grid_s, grid_s)

    sphere = get_sphere('repulsion724')

    if gtab:
        dsmodel = DiffusionSpectrumModel(gtab)
        signal_xq = np.fft.fftn(signal, axes=[-3, -2, -1])
        dsfit = dsmodel.fit(signal_xq.reshape(*s[:2],-1))
        odfs = dsfit.odf(sphere)
    else:
        signal = signal.reshape(-1, grid_s, grid_s, grid_s)
        odfs = np.stack([get_odf(signal[vid], sphere) for vid in range(signal.shape[0])]).reshape(*s[:-1], -1)

    if len(odfs.shape) == 3:
        odfs = odfs[:,:,np.newaxis]

    # visualize
    r = window.Scene()
    sfu = actor.odf_slicer(odfs, sphere=sphere, colormap='plasma', scale=0.5)
    sfu.display(z=0)
    r.add(sfu)
    window.show(r)

