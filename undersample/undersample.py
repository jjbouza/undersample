import numpy as np
from undersample.interpolate import *
import scipy.io as sio

from undersample.utils import visualize_pdf, gtable_from_qvecs

def undersample(S, mask, qvecs, method='sinc'):
    """
    Undersamples S using the mask. We assume S has the domain S(x,q) as provided in HCP by default, and the dimensions of 
    the mask must match the dimensions of S. We also assume S has two spatial dimensions (i.e. x is 2d) and three radial
    dimensions (i.e. q is 3d). We DO NOT assume S(q) is on a grid, so we support, e.g. multi-shell aquired S. For this reason
    we require a qvecs parameter to specify the locations of q samples. To map non-cartesian data to a grid we use n-linear interpolation.
    """

    # verify some basic conditions
    assert(qvecs.shape[0] == S.shape[-1])
    assert(len(mask.shape) == 5 or len(mask.shape) == 6)
    
    # do interpolation of Skq in q to generate a q-grid
    if method == 'linear':
        Sxq_grid, q_grid = interpolate_q_space_linear(S, qvecs, mask.shape[-1])
    if method == 'sinc':
        Sxq_grid, q_grid = interpolate_q_space_sinc(S, qvecs, mask.shape[-1])

    # start by doing a partial fft along the x dimensions of S
    if len(mask.shape) == 5:
        Skq = np.fft.fftn(Sxq_grid, axes=(0,1))
    if len(mask.shape) == 6:
        Skq = np.fft.fftn(Sxq_grid, axes=(0,1,2))

    # multiply by mask, done...
    undersampled_Skq = Skq.reshape(*Skq.shape[:-1], mask.shape[-1], mask.shape[-1], mask.shape[-1])*mask

    # now do inverse fft along x 
    if len(mask.shape) == 5:
        undersampled_Sxq = np.real(np.fft.ifftn(undersampled_Skq, axes=(0,1)))
    if len(mask.shape) == 6:
        undersampled_Sxq = np.real(np.fft.ifftn(undersampled_Skq, axes=(0,1,2)))


    return undersampled_Sxq, q_grid
    
# CLI for undersampling a saved patch
if __name__ == '__main__':
    from argparse import ArgumentParser
    from dipy.io.image import load_nifti
    from dipy.io.gradients import read_bvals_bvecs
    from undersample.generate_mask import generate_kq_power_density_mask
    import os

    parser = ArgumentParser(description='Undersample a single arbitrarly q-space sampled NIFTI image.')
    parser.add_argument('data_name', type=str, help='Filename of NIFTI data')
    parser.add_argument('bval_name', type=str, help='Filename of bval data')
    parser.add_argument('bvec_name', type=str, help='Filename of bvec data')

    parser.add_argument('--q_grid_size', type=int, default=12, help='Side length of interpolated q-space grid.')
    parser.add_argument('--undersample_pctg', type=float, default=0.5, help='Percentage (0<1) of data to keep during undersampling.')
    parser.add_argument('--q_radius', type=float, default=0.3, help='Normalized radius (0<1) to be fully sampled.')
    parser.add_argument('--p', type=int, default=4, help='Variable density sampling exponent.')

    parser.add_argument('--verbose', help='Print progress.', action='store_true')
    parser.add_argument('--mat_format', help='Save data in .mat format for usage with EAP_estimator_jess.', action='store_true')
    parser.add_argument('--visualize', help='Visualize the generated pdfs', action='store_true')

    args = parser.parse_args()

    def print_maybe(string):
        if args.verbose:
            print(string)

    print_maybe("Loading input.")
    data, affine = load_nifti(args.data_name)
    bvals, bvecs = read_bvals_bvecs(args.bval_name, args.bvec_name)

    dimsk = 2 if len(data.shape) == 3 else 3

    # qvecs generation
    qvecs = np.stack([bval*bvec for (bval, bvec) in zip(bvals, bvecs)])
    print_maybe("Generating Groundtruth")
    interpolated_gt = interpolate_q_space_sinc(data, qvecs, args.q_grid_size)[0].reshape(*data.shape[:-1], *([args.q_grid_size]*3))
    print_maybe("Generating mask")
    mask = generate_kq_power_density_mask(args.undersample_pctg, args.q_radius, dimsk, data.shape[0], args.q_grid_size, args.p)
    print_maybe("Undersampling")
    undersampled_slice, q_grid = undersample(data, mask, qvecs)
    
    # fourier transform of q dimensions followed by reshaping to get pdf vector at each location
    undersampled_slice_xr = np.real(np.fft.ifftn(undersampled_slice, axes=(-3,-2,-1)).reshape(*undersampled_slice.shape[:dimsk],-1))
    interpolated_gt_xr = np.real(np.fft.ifftn(interpolated_gt, axes=(-3,-2,-1)).reshape(*interpolated_gt.shape[:dimsk],-1))
    
    # normalize pdfs
    undersampled_pxr = undersampled_slice_xr/np.linalg.norm(undersampled_slice_xr, axis=-1)[:,:,None]
    groundtruth_pxr = interpolated_gt_xr/np.linalg.norm(interpolated_gt_xr, axis=-1)[:,:,None]
    print_maybe("Saving...")
    if args.mat_format:
        sio.savemat(os.path.splitext(args.data_name)[0]+'.mat', {'undersampled': undersampled_pxr, 'ground_truth': groundtruth_pxr, 'qvecs': qvecs, 'q_grid': q_grid} )
    else:
        np.savez(os.path.splitext(args.data_name)[0], undersampled=undersampled_pxr, ground_truth=groundtruth_pxr, qvecs=qvecs, q_grid=q_grid)

    if args.visualize:
        print("Undersampled P(x,r)")
        visualize_pdf(undersampled_pxr, gtable_from_qvecs(q_grid))
        print("Ground Truth P(x,r)")
        visualize_pdf(groundtruth_pxr, gtable_from_qvecs(q_grid))
    
    print_maybe("Done.")




