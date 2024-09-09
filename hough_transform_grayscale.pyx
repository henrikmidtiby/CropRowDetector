# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from cython.parallel cimport parallel, prange
import cython

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
    cdef cnp.ndarray[cnp.float64_t, ndim=2] accum
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bins
    cdef int max_distance, offset

    offset = int(ceil(sqrt(img.shape[0] * img.shape[0] +
                                   img.shape[1] * img.shape[1])))
    max_distance = 2 * offset + 1
    accum = np.zeros((max_distance, theta.shape[0]), dtype=np.float64)
    bins = np.linspace(-offset, offset, max_distance)

    # compute the nonzero indexes
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] x_idxs, y_idxs
    y_idxs, x_idxs = np.nonzero(img)

    # finally, run the transform
    cdef int nidxs, nthetas, x, y, value, accum_idx, i, j

    nidxs = y_idxs.shape[0]  # x and y are the same shape
    nthetas = theta.shape[0]


    #for kk in range(1):
    for j in prange(nthetas, nogil=True):
        for i in range(nidxs):
            x = x_idxs[i]
            y = y_idxs[i]
            value = <int>round((ctheta[j] * x + stheta[j] * y))
            accum_idx = value + offset
            accum[accum_idx, j] += 1

    return accum, theta, bins


"""
cpdef hough_line(cnp.ndarray img, cnp.ndarray[cnp.float64_t, ndim=1] theta):
    # Compute the array of angles and their sine and cosine
    # Can only be float64 because that is what np.sin and np.cos return
    cdef cnp.ndarray[cnp.float64_t, ndim=1] ctheta
    cdef cnp.ndarray[cnp.float64_t, ndim=1] stheta  
    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    cdef cnp.ndarray[cnp.uint8_t, ndim=2] img_temp
    img_temp = img.astype(np.uint8)
    
    # compute the bins and allocate the accumulator array
    cdef cnp.ndarray[cnp.uint32_t, ndim=2] accum, accum_running_avg # 32 bit int
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bins
    cdef Py_ssize_t max_distance, offset

    offset = <Py_ssize_t>ceil(sqrt(img.shape[0] * img.shape[0] +
                                   img.shape[1] * img.shape[1]))
    max_distance = 2 * offset + 1
    accum = np.zeros((max_distance, theta.shape[0]), dtype=np.uint32)
    bins = np.linspace(-offset, offset, max_distance)

    # compute the nonzero indexes
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] x_idxs, y_idxs
    y_idxs, x_idxs = np.nonzero(img)

    # finally, run the transform
    cdef Py_ssize_t nidxs, nthetas, i, j, x, y, accum_idx

    nidxs = y_idxs.shape[0]  # x and y are the same shape
    nthetas = theta.shape[0]
    with nogil:
        for i in range(nidxs):
            x = x_idxs[i]
            y = y_idxs[i]
            for j in range(nthetas):
                accum_idx = <Py_ssize_t>round(ctheta[j] * x + stheta[j] * y) + offset
                accum[accum_idx, j] += img_temp[y, x]

    accum_running_avg = accum.copy()
    # The accumulator array does not compute the average for the top and bottom 2 rows
    for i in range(2, accum.shape[0]-2):
        for j in range(0, accum.shape[1]):
            try:
                accum_running_avg[i,j] = (accum[i-2,j]+accum[i-1,j]+accum[i,j]+accum[i+1,j]+accum[i+2,j])/5
            except Exception as e:
                print(e)
    
    # Slows down by factor 4
    # accum_running_avg = running_avg(accum)

    cdef cnp.ndarray[cnp.float64_t, ndim=2] accum_float = accum_running_avg.astype(np.float64)

    return accum_float, theta, bins"""