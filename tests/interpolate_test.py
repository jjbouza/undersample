from interpolate import *
from utils import load_slice, gtable_from_qvecs, vis_2d_field

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.data import get_sphere

def oned_test():
    # non-uniform samples
    xi = np.sort(np.random.randn(20)).reshape(-1, 1)
    fxi = np.cos(np.sin(xi))**2
    # uniform samples
    zi = np.linspace(-1.5,1.5,20).reshape(-1, 1)

    # ground truth
    xgt = np.linspace(-1.5,1.5,100)
    fgt = np.cos(np.sin(xgt))**2

    # interpolate
    fzi = interpolate_sinc(xi, fxi, zi)

    # plot
    plt.plot(zi, fzi, label='interpolated')
    plt.plot(xi, fxi, 'ro')
    plt.plot(xgt, fgt, label='ground truth')
    plt.legend()
    plt.show()

def dipy_gt():
    print("Loading slice...")
    sample_slice, qvecs, gtab = load_slice('./test_data/david_data', '3112_BL_data_subject_space.nii.gz', gtab=True)
    mapmodel = GeneralizedQSamplingModel(gtab)
    mapfit = mapmodel.fit(sample_slice)
    sphere = get_sphere('repulsion724')
    odfs = mapfit.odf(sphere)
    vis_2d_field(odfs, sphere)

def dmri_test():
    print("Loading slice...")
    sample_slice, qvecs = load_slice('./test_data/david_data', '3112_BL_data_subject_space.nii.gz')

    # sinc interpolate
    Sxq_cc, qvecs_s = interpolate_q_space_sinc(sample_slice, qvecs, 12)

    gtab = gtable_from_qvecs(qvecs_s)

    # visualize tensor field using dipy
    dsmodel = DiffusionSpectrumModel(gtab)
    dsfit = dsmodel.fit(Sxq_cc)
    sphere = get_sphere('repulsion724')
    odfs = dsfit.odf(sphere)
    vis_2d_field(odfs, sphere)


if __name__ == "__main__":
    dmri_test()
    #dipy_gt()
