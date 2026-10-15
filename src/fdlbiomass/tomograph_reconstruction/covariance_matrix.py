import cupy as cp 
import numpy as np
from scipy import ndimage
import multiprocessing as mp
import pickle 
import os 
import pandas as pd 
import sys 
from scipy import ndimage
import pickle
import numpy as np 

def calculate_covariance_matrix_tutorial(img_stack, outname, kernelsize=10, overwrite=False):
    """
    compute the covariance matrix.
    If the target file already exists and ``overwrite=False`` this function acts as a simple file reader.

    Parameters
    ----------
    img_stack: numpy.ndarray
        the normalized SLC image stack
    outname: str
       the name of the file to be written
    kernelsize: int
        the boxcar smoothing dimension
    overwrite: bool
        overwrite an existing file? Otherwise it is read from file and returned

    Returns
    -------
    numpy.ndarray
        the covariance matrix
    """

    if overwrite or not os.path.isfile(outname):
        rows, cols, nTrack = img_stack.shape

        kernel = np.ones((kernelsize, kernelsize))
        weight = kernel / np.sum(kernel)

        cov_matrix = np.empty((rows, cols, nTrack, nTrack), np.complex64)
        smooth = np.empty((rows, cols, nTrack), np.float32)

        for i in range(0, nTrack):
            mm = np.absolute(img_stack[:, :, i]) ** 2
            smooth[:, :, i] = ndimage.convolve(mm, weight, mode='reflect')

        for i in range(0, nTrack):
            for j in range(0, nTrack):
                # multiply S1 and complex conjugate of S2
                ms = np.multiply(img_stack[:, :, i], np.conjugate(img_stack[:, :, j]))

                ms_real_smooth = ndimage.convolve(ms.real, weight, mode='reflect')
                ms_imag_smooth = complex(0, 1) * ndimage.convolve(ms.imag, weight, mode='reflect')
                ms_smooth = ms_real_smooth + ms_imag_smooth

                cov_matrix[:, :, i, j] = ms_smooth / (smooth[:, :, i] * smooth[:, :, j]) ** 0.5

        # save the variable
        with open(outname, 'wb') as f:
            pickle.dump(cov_matrix, f, 2)
    else:
        with open(outname, 'rb') as f:
            cov_matrix = pickle.load(f)
    return cov_matrix




def compare_matrices_detailed(original, optimized):
    """
    Compare two matrices (original and optimized) and calculate difference statistics.
    
    Args:
        original (cp.ndarray or np.ndarray): The original matrix from tutorial fn .
        optimized (cp.ndarray or np.ndarray): The optimized matrix to compare against the original.
    
    Returns:
        tuple: A tuple containing:
            - max_diff (float): The maximum absolute difference between the matrices.
            - mean_diff (float): The mean absolute difference between the matrices.
            - percentage_exceed (float): The percentage of elements that exceed a threshold difference.
    
    Raises:
        TypeError: If the input arrays are of an unsupported type.
    """
    threshold = 1e-6

    if isinstance(original, cp.ndarray):
        optimized = cp.array(optimized)
        diff = cp.abs(original - optimized)
        max_diff = float(cp.max(diff).get())
        mean_diff = float(cp.mean(diff).get())
        percentage_exceed = float((diff > threshold).sum().get()) / diff.size * 100
    elif isinstance(original, np.ndarray):
        diff = np.abs(original - optimized)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        percentage_exceed = (diff > threshold).sum() / diff.size * 100
    else:
        raise TypeError("Unsupported array type")

    return max_diff, mean_diff, percentage_exceed


