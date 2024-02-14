# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange

#from cpython.mem cimport PyMem_Malloc, PyMem_Free
#from libc.stdlib cimport labs
from libc.math cimport fabs, sqrt, ceil, atan2, M_PI, round

# DTYPE_t = cnp.float64_t

#cnp.import_array()

"""def hough_line_transform(np.ndarray[DTYPE_t, ndim=2] img, int theta_res=1, int rho_res=1):
    cdef int rows = img.shape[0]
    cdef int cols = img.shape[1]
    cdef int max_rho = int(np.ceil(np.sqrt(rows**2 + cols**2)))
    cdef int theta_size = int(180 / theta_res)
    cdef int rho_size = int(2 * max_rho / rho_res) + 1
    cdef np.ndarray[DTYPE_t, ndim=2] accumulator = np.zeros((rho_size, theta_size), dtype=DTYPE)

    cdef int x, y, rho, theta
    cdef float radian
    for y in range(rows):
        for x in range(cols):
            if img[y, x] > 0:
                for theta in range(theta_size):
                    radian = np.deg2rad(theta * theta_res)
                    rho = int(x * np.cos(radian) + y * np.sin(radian))
                    accumulator[rho + max_rho, theta] += 1

    return accumulator"""


cpdef hough_line(cnp.ndarray img, cnp.ndarray[cnp.float64_t, ndim=1] theta):
    # Compute the array of angles and their sine and cosine
    cdef cnp.ndarray[cnp.float64_t, ndim=1] ctheta
    cdef cnp.ndarray[cnp.float64_t, ndim=1] stheta

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    # compute the bins and allocate the accumulator array
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] accum
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bins
    cdef int max_distance, offset

    offset = int(ceil(sqrt(img.shape[0] * img.shape[0] +
                                   img.shape[1] * img.shape[1])))
    max_distance = 2 * offset + 1
    accum = np.zeros((max_distance, theta.shape[0]), dtype=np.uint64)
    bins = np.linspace(-offset, offset, max_distance)

    # compute the nonzero indexes
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] x_idxs, y_idxs
    y_idxs, x_idxs = np.nonzero(img)

    # finally, run the transform
    cdef int nidxs, nthetas, i, j, x, y, accum_idx, value

    nidxs = y_idxs.shape[0]  # x and y are the same shape
    nthetas = theta.shape[0]

    for kk in range(1):
        for j in prange(nthetas, nogil=True):
            for i in range(nidxs):
                x = x_idxs[i]
                y = y_idxs[i]
                value = <int>round((ctheta[j] * x + stheta[j] * y))
                accum_idx = value + offset
                accum[accum_idx, j] += 1

    return accum, theta, bins
