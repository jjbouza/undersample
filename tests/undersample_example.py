import numpy as np

from dipy.data import get_sphere
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel, DiffusionSpectrumModel

from undersample.generate_mask import generate_kq_power_density_mask
from undersample.undersample import undersample
from undersample.interpolate import interpolate_q_space_sinc
from undersample.utils import vis_2d_field, gtable_from_qvecs, load_slice


if __name__ == "__main__":
    print("Loading slice...")
    sample_slice, qvecs = load_slice('./test_data/T1w/Diffusion')
    interpolated_gt, _ = interpolate_q_space_sinc(sample_slice, qvecs, 12)
    print("Generating mask...")
    mask = generate_kq_power_density_mask(0.5, 0.6, 2, 36, 12, 4)
    print("Undersampling...")
    undersampled_slice, q_grid = undersample(sample_slice, mask, qvecs)
    print("Done undersampling!")

    print("Reconstructing using DSI")

    sphere = get_sphere('repulsion724')
    gtab = gtable_from_qvecs(q_grid)
    dsmodel = DiffusionSpectrumModel(gtab)
    dsfit = dsmodel.fit(undersampled_slice.reshape(*undersampled_slice.shape[0:2], -1))
    #dsfit = dsmodel.fit(interpolated_gt)
    odfs = dsfit.odf(sphere)
    vis_2d_field(odfs, sphere)
