I have been reading the Wenxing et. al. papers on multivariate sinc functions and their applications to dMRI
reconstruction. Here is the idea in the 2012 Transactions of Medical Imaging paper 
“An Efficient Multi-Shell Sampling Scheme for Reconstruction of Diffusion Propagators”:

* Do the q-space acquisition on some non-uniform points. 
* Use sinc interpolation to get the q-space samples on a lattice (e.g. Cartesian or BCC) 
* Apply the regular FFT to the interpolated lattice data to get the EAP.

So we need to do sinc interpolation onto a Cartessian lattice, undersample and then apply the FFT. This is a simple
underdetermined linear system.
