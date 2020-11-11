# python library for gridding, degridding, autocalibrated sensitivity map, and CG SENSE in 2D
#
# Code modified based on code from Nick Zwart at BNI
# author: Mike Schar

import numpy as np

def window2(shape, windowpct=100.0, widthpct=100.0, stopVal=0, passVal=1):
    # 2D hanning window just like shapes
    #   OUTPUT: 2D float32 circularly symmetric hanning

    import numpy as np

    # window function width
    bnd = 100.0/widthpct

    # generate coords for each dimension
    x = np.linspace(-bnd, bnd, shape[-1], endpoint=(shape[-1] % 2 != 0))
    y = np.linspace(-bnd, bnd, shape[-2], endpoint=(shape[-2] % 2 != 0))

    # create a 2D grid with coordinates then get radial coords
    xx, yy = np.meshgrid(x,y)
    radius = np.sqrt(xx*xx + yy*yy)

    # calculate hanning
    windIdx = radius <= 1.0
    passIdx = radius <= (1.0 - (windowpct/100.0))
    func = 0.5 * (1.0 - np.cos(np.pi * (1.0 - radius[windIdx]) / (windowpct/100.0)))

    # populate output array
    out = np.zeros(shape, dtype=np.float32)
    out[windIdx] = stopVal + func * (passVal - stopVal)
    out[passIdx] = passVal

    return out

# by Gabriele Bonanno
# July 2018

def window1(shape, windowpct=100.0, widthpct=100.0, stopVal=0, passVal=1):
    # 1D hanning window just like shapes
    #   OUTPUT: 1D float32 circularly symmetric hanning
    # TODO: add check on shape
    
    import numpy as np
    
    # window function width
    bnd = 100.0/widthpct
    
    # generate coords for each dimension
    x = np.linspace(-bnd, bnd, shape, endpoint=(shape % 2 != 0))
    
    # get radial coords
    radius = abs(x)
    
    # calculate hanning
    windIdx = radius <= 1.0
    passIdx = radius <= (1.0 - (windowpct/100.0))
    func = 0.5 * (1.0 - np.cos(np.pi * (1.0 - radius[windIdx]) / (windowpct/100.0)))
    
    # populate output array
    out = np.zeros(shape, dtype=np.float32)
    out[windIdx] = stopVal + func * (passVal - stopVal)
    out[passIdx] = passVal
    
    return out


def rolloff2D(mtx_xy, kernel, clamp_min_percent=5):
    # mtx_xy: int
    import numpy as np
    import triggeredGASSP.gridding.grid_kaiser as gd

    # grid one point at k_0
    dx = dy = 0.0
    coords = np.array([0,0], dtype='float32')
    data = np.array([1.0], dtype='complex64')
    weights = np.array([1.0], dtype='float32')
    outdim = np.array([mtx_xy, mtx_xy],dtype=np.int64)

    # grid -> fft -> |x|
    out = np.abs(fft2D(gd.grid(coords, data, weights, kernel, outdim, dx, dy)))

    # clamp the lowest values to a percentage of the max
    clamp = out.max() * clamp_min_percent/100.0
    out[out < clamp] = clamp

    # invert
    return 1.0/out

def kaiserbessel_kernel(kernel_table_size, oversampling_ratio):
    #   Generate a Kaiser-Bessel kernel function
    #   OUTPUT: 1D kernel table for radius squared

    import triggeredGASSP.gridding.grid_kaiser as dg
    kernel_dim = np.array([kernel_table_size],dtype=np.int64)
    return dg.kaiserbessel_kernel(kernel_dim, np.float64(oversampling_ratio))

def fft2D(data, dir=0, out_dims_fft=[]):
    # data: np.complex64
    # dir: int (0 or 1)
    # outdims = [nr_coils, extra_dim2, extra_dim1, mtx, mtx]

    import gpi_core.math.fft as corefft

    # generate output dim size array
    # fortran dimension ordering
    if len(out_dims_fft):
        outdims = out_dims_fft.copy()
    else:
        outdims = list(data.shape)
    
    outdims.reverse()
    outdims = np.array(outdims, dtype=np.int64)

    # load fft arguments
    kwargs = {}
    kwargs['dir'] = dir

    # transform
    kwargs['dim1'] = 1
    kwargs['dim2'] = 1
    kwargs['dim3'] = 0
    kwargs['dim4'] = 0
    kwargs['dim5'] = 0

    return corefft.fftw(data, outdims, **kwargs)

def grid2D(data, coords, weights, kernel, out_dims, number_threads=8):
    # data: np.float32
    # coords: np.complex64
    # weights: np.float32
    # kernel: np.float64
    # outdims = [nr_coils, extra_dim2, extra_dim1, mtx_xy, mtx_xy]: int
    import triggeredGASSP.gridding.grid_kaiser as bni_grid
    
    [nr_coils, extra_dim2, extra_dim1, mtx_xy, nr_arms, nr_points] = out_dims
    
    # threading is done along the coil dimension. Limit the number of threads to the number of coils:
    if number_threads > nr_coils:
        number_threads = nr_coils
    # make number_threads a divider of nr_coils:
    while (nr_coils%number_threads != 0):
        number_threads -= 1

    
    # off-center in pixels.
    dx = dy = 0.

    # gridded kspace
    gridded_kspace = np.zeros([nr_coils, extra_dim2, extra_dim1, mtx_xy, mtx_xy], dtype=data.dtype)
    
    # tell the grid routine what shape to produce
    outdim = np.array([mtx_xy,mtx_xy,nr_coils], dtype=np.int64)

    # coordinate dimensions
    same_coords_for_all_extra1 = False
    same_coords_for_all_extra2 = False
    if coords.ndim == 4:
        if coords.shape[0] == 1:
            same_coords_for_all_extra1 = True
    elif coords.ndim == 5:
        if coords.shape[1] == 1:
            same_coords_for_all_extra1 = True
        if coords.shape[0] == 1:
            same_coords_for_all_extra2 = True
    
    # grid all slices
    dx = dy = 0.
    for extra1 in range(extra_dim1):
        if same_coords_for_all_extra1:
            extra1_coords = 0
        else:
            extra1_coords = extra1
        for extra2 in range(extra_dim2):
            if same_coords_for_all_extra2:
                extra2_coords = 0
            else:
                extra2_coords = extra2
            if coords.ndim==5:   #DAN ZHU
                gridded_kspace[:,extra2,extra1,:,:] = bni_grid.grid(coords[extra2_coords,extra1_coords,:,:,:], data[:,extra2,extra1,:,:], weights[extra2_coords,extra1_coords,:,:], kernel, outdim, dx, dy, numThreads=number_threads)
            else:
                gridded_kspace[:,extra2,extra1,:,:] = bni_grid.grid(coords[extra1_coords,:,:,:], data[:,extra2,extra1,:,:], weights[extra1_coords,:,:], kernel, outdim, dx, dy, numThreads=number_threads)

    return gridded_kspace

def autocalibrationB1Maps2D(images, taper=50, width=10, mask_floor=1, average_csm=0):
    # dimensions
    mtx        = images.shape[-1]
    extra_dim1 = images.shape[-3]
    extra_dim2 = images.shape[-4]
    nr_coils   = images.shape[-5]

    # Dynamic data - average all dynamics for csm
    if ( (extra_dim1 > 1) and (average_csm) ):
        images_for_csm = images.sum(axis=-3)
        images_for_csm.shape = [nr_coils,extra_dim2,1,mtx,mtx]
        if ( (extra_dim2 > 1) and (average_csm) ):
            images_for_csm = images_for_csm.sum(axis=-4)
            images_for_csm.shape = [nr_coils,1,1,mtx,mtx]
    else:
        images_for_csm = images

    # generate window function for blurring image data
    win = window2(images_for_csm.shape[-2:], windowpct=taper, widthpct=width)

    # apply kspace filter
    kspace = fft2D(images_for_csm, dir=1)
    kspace *= win

    # transform back into image space and normalize
    csm = fft2D(kspace, dir=0)
    rms = np.sqrt(np.sum(np.abs(csm)**2, axis=0))
    csm = csm / rms

    # zero out points that are below the mask threshold
    thresh = mask_floor/100.0 * rms.max()
    csm *= rms > thresh
    
    # Dynamic data - average all dynamics for csm - asign the average to all dynamics
    if ( (extra_dim1 > 1) and (average_csm) ):
        out = np.zeros(images.shape, np.complex64)
        for coil in range(nr_coils):
            for extra2 in range(extra_dim2):
                for extra1 in range(extra_dim1):
                    out[coil,extra2,extra1,:,:] = csm[coil,0,0,:,:]
    else:
        out=csm

    return out

def degrid2D(data, coords, kernel, outdims, number_threads=8, oversampling_ratio=1.):
    # data: np.float32
    # coords: np.complex64
    # weights: np.float32
    # kernel: np.float64
    # outdims = [nr_coils, extra_dim2, extra_dim1, mtx_xy, mtx_xy]: int
    # number_threads: int
    import triggeredGASSP.gridding.grid_kaiser as bni_grid
    
    [nr_coils, extra_dim2, extra_dim1, nr_arms, nr_points] = outdims
    
    # coordinate dimensions
    same_coords_for_all_extra1 = False
    same_coords_for_all_extra2 = False
    if coords.ndim == 4:
        if coords.shape[0] == 1:
            same_coords_for_all_extra1 = True
    elif coords.ndim == 5:
        if coords.shape[1] == 1:
            same_coords_for_all_extra1 = True
        if coords.shape[0] == 1:
            same_coords_for_all_extra2 = True


    # gridded kspace
    degridded_kspace = np.zeros([nr_coils, extra_dim2, extra_dim1, nr_arms, nr_points], dtype=data.dtype)

    # degrid all slices
    for extra1 in range(extra_dim1):
        if same_coords_for_all_extra1:
            extra1_coords = 0
        else:
            extra1_coords = extra1
        for extra2 in range(extra_dim2):
            if same_coords_for_all_extra2:
                extra2_coords = 0
            else:
                extra2_coords = extra2
            for coil in range(nr_coils):
                if coords.ndim==5:
                    degridded_kspace[coil,extra2,extra1,:,:] = bni_grid.degrid(coords[extra2_coords,extra1_coords,:,:,:], data[coil,extra2,extra1,:,:], kernel, number_threads) * oversampling_ratio * oversampling_ratio
                else:
                    degridded_kspace[coil,extra2,extra1,:,:] = bni_grid.degrid(coords[extra1_coords,:,:,:], data[coil,extra2,extra1,:,:], kernel, number_threads) * oversampling_ratio * oversampling_ratio

    return degridded_kspace

