# Copyright (c)
# Author: Mike Schär
# Date: 2017 July 28

import numpy as np
import cupy as cp
import gpi
import sys

class ExternalNode(gpi.NodeAPI):
    """2D Non-Linear CG reconstruction.
      
    * Feng L, Grimm R, Tobias Block K, Chandarana H, Kim S, Xu J, Axel L, 
      Sodickson DK, Otazo R. "Golden-angle radial sparse parallel MRI: 
      Combination of compressed sensing, parallel imaging, and golden-angle 
      radial sampling for fast and flexible dynamic volumetric MRI." 
      Magn Reson Med. 2013 Oct 18. doi: 10.1002/mrm.24980
        
    * Lustig M, Donoho D, Pauly JM. "Sparse MRI: The application of compressed 
      sensing for rapid MR imaging." Magn Reson Med. 2007 Dec;58(6):1182-95.

    * Pruessmann, Klaas P., et al. "Advances in sensitivity encoding with
      arbitrary k‐space trajectories." Magnetic Resonance in Medicine 46.4
      (2001): 638-651.
    * Shewchuk, Jonathan Richard. "An introduction to the conjugate gradient
      method without the agonizing pain." (1994).

    WIDGETS:
        mtx: the matrix to be used for gridding (this is the size used no extra scaling is added)
        iterations: number of iterations to complete before terminating
        lamb: lambda, regularization factor to weight data consistency vs. temporal total variation
        oversampling ratio: oversampling factor, default 1.375 with kernel size of 5 according to Beaty et al.
        number of threads: set accroding to number of CPUs
        Autocalibration Width (%): percentage of pixels to use for B1 est.
        Autocalibration Taper (%): han window taper for blurring.
        Mask Floor (% of max mag): Mask the coil sensitivity map
        Autocalibration mask dilation [pixels]: dilate the mask
        Autocalibration SDC Iterations: number of iterations for SDC calculation of coil sensitivity maps (csm)
        Dynamic data - average all dynamics before gridding: if dynamics are aquired with the same trajectory
        Golden Angle - combine dynamics before gridding: Combine all data and use the same for all dynamics
        Store iterations: store all iterations for debugging, requires significant memory

    INPUT:
        data: raw k-space data
        coords: trajectory coordinates scaled from -0.5 to 0.5
        weights: sample density weights for gridding
        coil sensitivity: non-conjugated sensitivity maps

    OUTPUT:
        x: solution at the current iteration
        Autocalibration CSM: B1-recv estimated using the central k-space points, oversampled
        Autocalibration CSM: B1-recv estimated using the central k-space points, cropped to selected FOV
        solutions for all iterations, may be using too much memory
        cost vs. time
    """

    def initUI(self):
        # Widgets
        self.addWidget('DoubleSpinBox', 'mtx', val=300, min=1)
        self.addWidget('SpinBox', 'iterations', val=10, min=1)
        self.addWidget('DoubleSpinBox', 'lambda TV cardiac dimension', val=0.01, min=0, max=1, decimals=6)
        self.addWidget('DoubleSpinBox', 'lambda TV flow dimension', val=0.015, min=0, max=1, decimals=6)
        self.addWidget('DoubleSpinBox', 'oversampling ratio', val=1.375, decimals=3, singlestep=0.125, min=1, max=2, collapsed=True)
        self.addWidget('SpinBox', 'number of threads', val=8, min=1, max=64, collapsed=True)
        self.addWidget('Slider', 'Autocalibration Width (%)', val=10, min=0, max=100)
        self.addWidget('Slider', 'Autocalibration Taper (%)', val=50, min=0, max=100)
        self.addWidget('Slider', 'Mask Floor (% of max mag)', val=10, min=0, max=100)
        self.addWidget('SpinBox', 'Autocalibration mask dilation [pixels]', val=10, min=0, max=100)
        self.addWidget('SpinBox','Autocalibration SDC Iterations',val=10, min=1)
        self.addWidget('PushButton', 'Dynamic data - average all dynamics for csm', toggle=True, button_title='ON', val=0)
        self.addWidget('PushButton', 'Golden Angle - combine dynamics before gridding', toggle=True, button_title='ON', val=1)
        self.addWidget('Slider', '# golden angle dynamics for csm', val=150, min=0, max=10000)
        self.addWidget('PushButton', 'Store iterations', toggle=True, button_title='ON', val=0)
    
        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=[np.complex64, np.complex128])
        self.addInPort('coords', 'NPYarray', dtype=[np.float32, np.float64])
        self.addInPort('weights', 'NPYarray', dtype=[np.float32, np.float64])
        self.addInPort('coil sensitivity', 'NPYarray', dtype=[np.complex64, np.complex128], obligation=gpi.OPTIONAL)
        
        self.addOutPort('out', 'NPYarray', dtype=np.complex64)
        self.addOutPort('oversampled CSM', 'NPYarray', dtype=np.complex64)
        self.addOutPort('cropped CSM', 'NPYarray', dtype=np.complex64)
        self.addOutPort('x iterations', 'NPYarray', dtype=np.complex64)
        self.addOutPort('cost vs. time', 'NPYarray', dtype=np.float32)

    def validate(self):
        self.log.debug("validate NonLinCG-balancedPenaltyFlow_GPU")
        
        # check size of data vs. coords
        self.log.debug("validate NonLinCG-balancedPenaltyFlow_GPU - check size of data vs. coords")
        data = self.getData('data')
        coords = self.getData('coords')
        if coords.shape[-1] != 2:
            self.log.warn("Currently only for 2D data")
            return 1
        if coords.shape[-2] != data.shape[-1]:
            self.log.warn("data and coords do not agree in the number of sampled points per arm")
            return 1
        if coords.shape[-3] != data.shape[-2]:
            self.log.warn("data and coords do not agree in the number of arms")
            return 1
        if coords.ndim == 4:
            if data.ndim < 4:
                self.log.warn("if coords has 4 dimensions then data also needs 4 or more dimensions")
                return 1
            else:
                if coords.shape[-4] != data.shape[-3]:
                    self.log.warn("data and coords do not agree in the number of phases / dynamics")
                    return 1

        # make sure this is phase contrast flow data
        if data.shape[1] != 2:
            self.log.warn("data 2nd dimension is not equal to 2 as expected for phase contrast data.")
            return 1

        self.log.debug("validate NonLinCG-balancedPenaltyFlow_GPU - check csm")
        csm = self.getData('coil sensitivity')
        if csm is None:
            self.setAttr('Autocalibration Width (%)', visible=True)
            self.setAttr('Autocalibration Taper (%)', visible=True)
            self.setAttr('Mask Floor (% of max mag)', visible=True)
            self.setAttr('Autocalibration mask dilation [pixels]', visible=True)
            self.setAttr('Autocalibration SDC Iterations', visible=True)
            if self.getData('data').ndim > 3:
                self.setAttr('Dynamic data - average all dynamics for csm', visible=True)
            else:
                self.setAttr('Dynamic data - average all dynamics for csm', visible=False)
        else:
            self.setAttr('Autocalibration Width (%)', visible=False)
            self.setAttr('Autocalibration Taper (%)', visible=False)
            self.setAttr('Mask Floor (% of max mag)', visible=False)
            self.setAttr('Autocalibration mask dilation [pixels]', visible=False)
            self.setAttr('Dynamic data - average all dynamics for csm', visible=False)
            self.setAttr('Autocalibration SDC Iterations', visible=False)
            
        GA = self.getVal('Golden Angle - combine dynamics before gridding')
        if GA:
            self.setAttr('# golden angle dynamics for csm', visible=True)
            if coords.ndim == 3:
                self.setAttr('# golden angle dynamics for csm', max=coords.shape[-3])
            elif coords.ndim == 4:
                self.setAttr('# golden angle dynamics for csm', max=coords.shape[-3]*coords.shape[-4])
        else:
            self.setAttr('# golden angle dynamics for csm', visible=False)
                    
        return 0
    

    def compute(self):
        import triggeredGASSP.gridding.Kaiser2D_utils as kaiser2D
        import time
        time_0 = time.time()
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        self.log.debug("GPU memory info - used: " + str(mempool.used_bytes()) + ", total: " + str(mempool.total_bytes()) + " pinned :" + str(pinned_mempool.n_free_blocks()))
        
        self.log.node("Start Non-Linear conjugate gradient algorithm in 2D")
        # get port and widget inputs
        data = self.getData('data').astype(np.complex64, copy=False)
        coords = self.getData('coords').astype(np.float32, copy=False)
        weights = self.getData('weights').astype(np.float32, copy=False)
        
        mtx_original = np.int( self.getVal('mtx') )
        iterations = self.getVal('iterations')
        lamb = self.getVal('lambda TV cardiac dimension')
        lamb2 = self.getVal('lambda TV flow dimension')
        oversampling_ratio = self.getVal('oversampling ratio')
        number_threads = self.getVal('number of threads')
        GA = self.getVal('Golden Angle - combine dynamics before gridding')
        store_iterations = self.getVal('Store iterations')
        
        csm = self.getData('coil sensitivity')
        if csm is not None:
            csm = csm.astype(np.complex64, copy=False)
        
        # oversampling: Oversample at the beginning and crop at the end
        mtx = np.int(np.around(mtx_original * oversampling_ratio))
        if mtx%2:
            mtx+=1
        if oversampling_ratio > 1:
            mtx_min = np.int(np.around((mtx-mtx_original)/2))
            mtx_max = mtx_min + mtx_original
        else:
            mtx_min = 0
            mtx_max = mtx
                
        # data dimensions
        nr_points = data.shape[-1]
        nr_arms = data.shape[-2]
        nr_coils = data.shape[0]
        if data.ndim == 3:
            extra_dim1 = 1
            extra_dim2 = 1
            data.shape = [nr_coils,extra_dim2,extra_dim1,nr_arms,nr_points]
        elif data.ndim == 4:
            extra_dim1 = data.shape[-3]
            extra_dim2 = 1
            data.shape = [nr_coils,extra_dim2,extra_dim1,nr_arms,nr_points]
        elif data.ndim == 5:
            extra_dim1 = data.shape[-3]
            extra_dim2 = data.shape[-4]
            #self.log.warn("what is extra_dim2? not implemented for new way of doing CSM.")
        elif data.ndim > 5:
            self.log.warn("Not implemented yet")
        out_dims_grid = [nr_coils, extra_dim2, extra_dim1, mtx, nr_arms, nr_points]
        out_dims_grid_per_flow = [nr_coils, 1, extra_dim1, mtx, nr_arms, nr_points]
        out_dims_degrid = [nr_coils, extra_dim2, extra_dim1, nr_arms, nr_points]
        out_dims_degrid_per_flow = [nr_coils, 1, extra_dim1, nr_arms, nr_points]
        out_dims_fft = [nr_coils, extra_dim2, extra_dim1, mtx, mtx]
        dims_L2Grad = [extra_dim2, extra_dim1, mtx, mtx]
        dims_L2Grad_per_flow = [nr_coils, 1, extra_dim1, mtx, mtx]
        iterations_shape = [extra_dim2, extra_dim1, mtx, mtx]
        
        data_gpu = cp.asarray(data)

        # coords dimensions: (add 1 dimension as they could have another dimension for golden angle dynamics
        if coords.ndim == 3:
            coords.shape = [1,1,nr_arms,nr_points,2]
            weights.shape = [1,1,nr_arms,nr_points]
        elif coords.ndim == 4:
            coords.shape = [1,extra_dim1,nr_arms,nr_points,2]
            weights.shape = [1,extra_dim1,nr_arms,nr_points]

        # output including all iterations
        if store_iterations:
            x_iterations = np.zeros([iterations+1,extra_dim2,extra_dim1,mtx_original,mtx_original],dtype=np.complex64)
        
        # pre-calculate Kaiser-Bessel kernel
        self.log.debug("Calculate kernel")
        kernel_table_size = 800
        kernel = kaiser2D.kaiserbessel_kernel( kernel_table_size, oversampling_ratio)

        # pre-calculate the rolloff for the spatial domain
        roll = kaiser2D.rolloff2D(mtx, kernel)
        roll_gpu = cp.asarray(roll)

        # scale SDC to make sure that Grid - DeGrid gets approximatly the same result
        scale_SDC = True
        if scale_SDC:
            for scale_SDC_counter1 in range(weights.shape[0]):
                for scale_SDC_counter2 in range(weights.shape[1]):
                    scale_factor = np.max(weights[scale_SDC_counter1,scale_SDC_counter2,:,(int)(0.85*nr_points):(int)(0.95*nr_points)])
                    self.log.debug("SDC scaling for extra_dim1 "+str(scale_SDC_counter1)+" extra_dim2 "+str(scale_SDC_counter2)+" = "+str(scale_factor))
                    weights[scale_SDC_counter1,scale_SDC_counter2,...] /= scale_factor
    
        if GA: #combine data from GA dynamics before gridding, use code from VirtualChannels_GPI.py
            # grid images for each phase - needs to be done at some point, not really here for csm though.
            self.log.debug("Grid undersampled data")
            #gpu use logically wrong name gridded_kspace_gpu = cp.asarray( kaiser2D.grid2D(data, coords, weights, kernel, out_dims_grid, number_threads=number_threads) )
            image_domain_gpu = cp.asarray( kaiser2D.grid2D(data, coords, weights, kernel, out_dims_grid, number_threads=number_threads) )

            # FFT
            for coil_idx in range(nr_coils):
                image_domain_gpu[coil_idx,...] = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(image_domain_gpu[coil_idx,...], axes=(-1,-2)), axes=(-1,-2)), axes=(-1,-2))

            # rolloff
            image_domain_gpu *= roll_gpu
        
            if csm is None:
                self.log.debug("Generating autocalibrated B1 maps...")
                twoD_or_threeD = coords.shape[-1]
                # parameters from UI
                UI_width = self.getVal('Autocalibration Width (%)')
                UI_taper = self.getVal('Autocalibration Taper (%)')
                UI_mask_floor = self.getVal('Mask Floor (% of max mag)')
                mask_dilation = self.getVal('Autocalibration mask dilation [pixels]')
                UI_average_csm = self.getVal('Dynamic data - average all dynamics for csm')
                numiter = self.getVal('Autocalibration SDC Iterations')
                original_csm_mtx = np.int(0.01 * UI_width * mtx_original)
                    
                is_GoldenAngle_data = True
                nr_arms_csm = self.getVal('# golden angle dynamics for csm')
                csm_data = data.copy()
                nr_all_arms_csm = extra_dim1 * nr_arms
                extra_dim1_csm = 1
                extra_dim2_csm = 1
                
                # coords dimensions: (add 1 dimension as they could have another dimension for golden angle dynamics
                if coords.ndim == 3:
                    coords.shape = [1,nr_arms,nr_points,twoD_or_threeD]
                    
                # create low resolution csm
                # cropping the data will make gridding and FFT much faster
                magnitude_one_interleave = np.zeros(nr_points)
                for x in range(nr_points):
                    magnitude_one_interleave[x] = np.sqrt( coords[0,0,0,x,0]**2 + coords[0,0,0,x,1]**2)
                within_csm_width_radius = magnitude_one_interleave[:] < (0.01 * UI_width * 0.5) # for BNI spirals should be 0.45 instead of 0.5
                nr_points_csm_width = within_csm_width_radius.sum()
                # in case of radial trajectory, it doesn't start at zero..
                found_start_point = 0
                found_end_point = 0
                for x in range(nr_points):
                    if ((not found_start_point) and (within_csm_width_radius[x])):
                        found_start_point = 1
                        start_point = x
                    if ((not found_end_point) and (found_start_point) and (not within_csm_width_radius[x])):
                        found_end_point = 1
                        end_point = x
                if not found_end_point:
                    end_point = nr_points
                self.log.node("Start and end points in interleave are: "+str(start_point)+" and "+str(end_point)+" leading to "+str(nr_points_csm_width)+" points for csm.")
                
                arm_counter = 0
                extra_dim1_counter = 0
                arm_with_data_counter = 0
                while (arm_with_data_counter < nr_arms_csm and extra_dim1_counter < extra_dim1):
                    if (coords[0,extra_dim1_counter, arm_counter,0,0] != coords[0,extra_dim1_counter, arm_counter,-1,0]): #only equal when no data in this interleave during resorting
                        arm_with_data_counter += 1
                    arm_counter += 1
                    if arm_counter == nr_arms:
                        arm_counter = 0
                        extra_dim1_counter += 1
                self.log.node("Found "+str(arm_with_data_counter)+" arms, and was looking for "+str(nr_arms_csm)+" from a total of "+str(nr_all_arms_csm)+" arms.")
                
                csm_data = np.zeros([nr_coils,1,extra_dim1_csm,arm_with_data_counter,nr_points_csm_width], dtype=data.dtype)
                csm_coords = np.zeros([1,arm_with_data_counter,nr_points_csm_width,twoD_or_threeD], dtype=coords.dtype)
                
                arm_counter = 0
                extra_dim1_counter = 0
                arm_with_data_counter = 0
                while (arm_with_data_counter < nr_arms_csm and extra_dim1_counter < extra_dim1):
                    if (coords[0,extra_dim1_counter, arm_counter,0,0] != coords[0,extra_dim1_counter, arm_counter,-1,0]): #only equal when no data in this interleave during resorting
                        csm_data[:,0,0,arm_with_data_counter,:] = data[:,0,extra_dim1_counter,arm_counter,start_point:end_point]
                        csm_coords[0,arm_with_data_counter,:,:] = coords[0,extra_dim1_counter,arm_counter,start_point:end_point,:]
                        arm_with_data_counter += 1
                    arm_counter += 1
                    if arm_counter == nr_arms:
                        arm_counter = 0
                        extra_dim1_counter += 1
                self.log.node("Found "+str(arm_with_data_counter)+" arms, and was looking for "+str(nr_arms_csm)+" from a total of "+str(nr_all_arms_csm)+" arms.")
                
                # now set the dimension lists
                out_dims_grid_csm = [nr_coils, extra_dim2_csm, extra_dim1_csm, mtx, arm_with_data_counter, nr_points_csm_width]
                out_dims_fft_csm = [nr_coils, extra_dim2_csm, extra_dim1_csm, mtx, mtx]
                
                # generate SDC based on number of arms and nr of points being used for csm
                import gpi_core.gridding.sdc as sd
                #csm_weights = sd.twod_sdcsp(csm_coords.squeeze().astype(np.float64), numiter, 0.01 * UI_taper, mtx)
                cmtxdim = np.array([mtx,mtx],dtype=np.int64)
                wates = np.ones((arm_with_data_counter * nr_points_csm_width), dtype=np.float64)
                coords_for_sdc = csm_coords.astype(np.float64)
                coords_for_sdc.shape = [arm_with_data_counter * nr_points_csm_width, twoD_or_threeD]
                csm_weights = sd.twod_sdc(coords_for_sdc, wates, cmtxdim, numiter, 0.01 * UI_taper )
                csm_weights.shape = [1,arm_with_data_counter,nr_points_csm_width]
                
                # Grid
                gridded_kspace_csm = kaiser2D.grid2D(csm_data, csm_coords, csm_weights.astype(np.float32), kernel, out_dims_grid_csm, number_threads=number_threads)
                image_domain_csm = kaiser2D.fft2D(gridded_kspace_csm, dir=0, out_dims_fft=out_dims_fft_csm)
                # rolloff
                image_domain_csm *= roll
                # normalize by rms (better would be to use a whole body coil image
                csm_rms = np.sqrt(np.sum(np.abs(image_domain_csm)**2, axis=0))
                image_domain_csm = image_domain_csm / csm_rms
                # zero out points that are below mask threshold
                thresh = 0.01 * UI_mask_floor * csm_rms.max()
                mask = csm_rms > thresh
                # use scipy library to grow mask and fill holes.
                from scipy import ndimage
                mask = ndimage.morphology.binary_dilation(mask, iterations=mask_dilation)
                mask = ndimage.binary_fill_holes(mask)
            
                image_domain_csm *= mask
                # extra_dim2/extra_dim1 dimensions are for flow encode/cardiac phases. Force them to size 1 and use FOR loops below to save memory
                csm = image_domain_csm
                
                # normalize csm
                csm /= np.max(np.abs(csm))
            else:
                # make sure input csm and data are the same mtx size.
                # Assuming the FOV was the same: zero-fill in k-space
                if csm.ndim != 5:
                    self.log.debug("Reshape imported csm")
                    csm.shape = [nr_coils,extra_dim2,extra_dim1,csm.shape[-2],csm.shape[-1]]
                if csm.shape[-1] != mtx:
                    self.log.debug("Interpolate csm to oversampled matrix size")
                    csm_oversampled_mtx = np.int(csm.shape[-1] * oversampling_ratio)
                    if csm_oversampled_mtx%2:
                        csm_oversampled_mtx+=1
                    out_dims_oversampled_image_domain = [nr_coils, extra_dim2, extra_dim1, csm_oversampled_mtx, csm_oversampled_mtx]
                    csm = kaiser2D.fft2D(csm, dir=1, out_dims_fft=out_dims_oversampled_image_domain)
                    csm = kaiser2D.fft2D(csm, dir=0, out_dims_fft=out_dims_fft)

            self.setData('oversampled CSM', csm)
            self.setData('cropped CSM', csm[...,mtx_min:mtx_max,mtx_min:mtx_max])
    
                    
        else: # this is the normal path (not GA)
            # grid to create images that are corrupted by
            # aliasing due to undersampling.  If the k-space data have an
            # auto-calibration region, then this can be used to generate B1 maps.
            self.log.debug("Grid undersampled data")
            gridded_kspace = kaiser2D.grid2D(data, coords, weights, kernel, out_dims_grid, number_threads=number_threads)
            # FFT
            image_domain = kaiser2D.fft2D(gridded_kspace, dir=0, out_dims_fft=out_dims_fft)
            # rolloff
            image_domain *= roll

            # calculate auto-calibration B1 maps
            if csm is None:
                self.log.debug("Generating autocalibrated B1 maps...")
                # parameters from UI
                UI_width = self.getVal('Autocalibration Width (%)')
                UI_taper = self.getVal('Autocalibration Taper (%)')
                UI_mask_floor = self.getVal('Mask Floor (% of max mag)')
                UI_average_csm = self.getVal('Dynamic data - average all dynamics for csm')
                csm = kaiser2D.autocalibrationB1Maps2D(image_domain, taper=UI_taper, width=UI_width, mask_floor=UI_mask_floor, average_csm=UI_average_csm)
            else:
                # make sure input csm and data are the same mtx size.
                # Assuming the FOV was the same: zero-fill in k-space
                if csm.ndim != 5:
                    self.log.debug("Reshape imported csm")
                    csm.shape = [nr_coils,extra_dim2,extra_dim1,csm.shape[-2],csm.shape[-1]]
                if csm.shape[-1] != mtx:
                    self.log.debug("Interpolate csm to oversampled matrix size")
                    csm_oversampled_mtx = np.int(csm.shape[-1] * oversampling_ratio)
                    if csm_oversampled_mtx%2:
                        csm_oversampled_mtx+=1
                    out_dims_oversampled_image_domain = [nr_coils, extra_dim2, extra_dim1, csm_oversampled_mtx, csm_oversampled_mtx]
                    csm = kaiser2D.fft2D(csm, dir=1, out_dims_fft=out_dims_oversampled_image_domain)
                    csm = kaiser2D.fft2D(csm, dir=0, out_dims_fft=out_dims_fft)
            self.setData('oversampled CSM', csm)
            self.setData('cropped CSM', csm[...,mtx_min:mtx_max,mtx_min:mtx_max])


        csm_gpu = cp.asarray(csm)
        # remove pinned memory as those block sizes will not be used again.
        pinned_mempool.free_all_blocks()
        #keep a conjugate csm set on hand
        # don't understand why, but cp.conj does not lead to a real number when cp.conj(a) * a
        csm_conj_gpu = cp.conj(csm_gpu)

        # Line search parameters (from GRASP code by Ricardo Otazo, NYU 2008
        maxlsiter = 12 #20 #150
        gradToll = 1e-3
        l1Smooth = 1e-15
        # pre balanced alpha = 0.01
        alpha = 0.25
        #alpha = 0.001
        beta = 0.5 #0.6
        t0 = 1.
        #k = 0
        eps = np.finfo( image_domain_gpu.dtype ).eps
        self.log.debug("eps = " + str(eps) + ", and 1./l1Smooth = " + str(1./l1Smooth))
        reset_dx_iter = 5
        # for debugging store cost and time
        # cost_time_array[ iterations, 2 ] with time and cost
        cost_time_array = np.zeros([ iterations, 2 ], dtype=np.float32)

        # calculate initial conditions
        for flow_idx in range(extra_dim2):
            for phase_idx in range(extra_dim1):
                image_domain_gpu[:,flow_idx,phase_idx,:,:] = csm_conj_gpu[:,0,0,:,:] * image_domain_gpu[:,flow_idx,phase_idx,:,:] # remove coil phase
        x_0_gpu = image_domain_gpu.sum(axis=0) # assume the coil dim is the first
        image_domain_gpu = None
        Lambda = lamb * cp.asnumpy(cp.max( cp.abs( x_0_gpu ) ))
        Lambda2 = lamb2 * cp.asnumpy(cp.max( cp.abs( x_0_gpu ) ))
        self.log.info("Lambda (cardiac) = "+str(Lambda) + " and Lambda2 (flow) = "+str(Lambda2))
        
        current_iteration = cp.asnumpy(x_0_gpu)
        current_iteration.shape = iterations_shape
        if store_iterations:
            x_iterations[0,:,:,:,:] = current_iteration[...,mtx_min:mtx_max,mtx_min:mtx_max]
        
        # during iterations, the objective is re-calculated in the process
        # when line-search is finished, the gradient is calculated for the next iteation. The gradient calculation can use some values from the last objective calculation.
        # Therefore, here, first the objective is calculated, and the gradient will be calculated in the first step within the iteration loop.

        # L2-norm part  performed on each channel
        L2Grad_degrid_gpu = cp.zeros(out_dims_degrid, dtype=np.complex64)
        for flow_idx in range(extra_dim2):
            L2Grad_per_flow_gpu = cp.zeros(dims_L2Grad_per_flow, dtype=np.complex64)
            for phase_idx in range(extra_dim1):
                L2Grad_per_flow_gpu[:,0,phase_idx,:,:] = csm_gpu[:,0,0,:,:] * x_0_gpu[flow_idx,phase_idx,:,:] # add coil phase
            L2Grad_per_flow_gpu *= roll_gpu # pre-rolloff for degrid convolution
            # want to do this but runs out of memory L2Grad = cp.asnumpy(cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(L2Grad_gpu), axes=(-1,-2))))
            for coil_idx in range(nr_coils):
                L2Grad_per_flow_gpu[coil_idx,...] = cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(L2Grad_per_flow_gpu[coil_idx,...], axes=(-1,-2)), axes=(-1,-2)), axes=(-1,-2))
            L2Grad_degrid_gpu[:,flow_idx:flow_idx+1,...] = cp.asarray( kaiser2D.degrid2D(cp.asnumpy(L2Grad_per_flow_gpu), coords[flow_idx:flow_idx+1,...], kernel, out_dims_degrid_per_flow, number_threads=number_threads, oversampling_ratio = oversampling_ratio) )
            L2Grad_per_flow_gpu = None
        self.log.debug("max degridded image = "+str(cp.abs(L2Grad_degrid_gpu).max())+" vs max of data = "+str(cp.abs(data_gpu).max()) )
        L2Grad_degrid_gpu -= data_gpu
        L2Obj = cp.asnumpy(cp.sum( cp.square( (cp.abs(L2Grad_degrid_gpu))))).astype(np.float32)

        # L1-norm cardiac part  performed on channel combined data
        w_card_gpu = x_0_gpu.copy()
        w_card_gpu = w_card_gpu[:, 1 + np.array( list(range(w_card_gpu.shape[1]-1)) + [-1] ) ,:,:] - w_card_gpu
        L1Obj_card = cp.asnumpy( cp.sum( cp.abs( w_card_gpu ) ) )

        # L1-norm flow part   only compare magnitude between flow encodes and keep phase (velocity) unchanged
        m_flow_gpu = cp.abs(x_0_gpu)
        m_flow_gpu = m_flow_gpu[np.array( [1,0] ) ,:,:,:] - m_flow_gpu
        L1Obj_flow = cp.asnumpy( cp.sum( cp.abs( m_flow_gpu ) ) )

        # before balancing penalty f0 = L2Obj + ( Lambda * L1Obj )\
        # seems the scaling between objective and gradient are very different, which leads to the effect that when using the same lambda for scaling, either component has no effect in either step.
        # Therefore, here we weight each component as 1 for the objective, and use the lambda scaling factor to weight the relative contributions to the gradient.
        f0 = 3.0
        f1 = 3.0
        self.log.info("Non-lin CG Flow:  Initial condition - prior to line search, f0 = " + str( f0)  + ", t = 0, L2Obj = " + str( L2Obj ) + ", L1Obj_card * Lambda = " + str( Lambda * L1Obj_card ) + ", L1Obj_flow * Lambda2 = " + str( Lambda2 * L1Obj_flow ))

        # initial objectives
        initial_L2 = L2Obj.copy()
        initial_L1_card = L1Obj_card.copy()
        initial_L1_flow = L1Obj_flow.copy()
        initial_f0 = f0

        # remove pinned memory as those block sizes will not be used again.
        pinned_mempool.free_all_blocks()
        self.log.debug("GPU memory info - prior to iterating - used: " + str(mempool.used_bytes()) + ", total: " + str(mempool.total_bytes()) + " pinned :" + str(pinned_mempool.n_free_blocks()) + "." )

        # conjugate gradient calculation with backtracking line search
        for i in range(iterations):
            # determine the gradient

            # Do L1-norm flow part first to be able to delete m_flow_gpu to reduce memory for L2 calculations
            # L1-norm flow part   only compare magnitude between flow encodes and keep phase (velocity) unchanged
            L1Grad_m_flow_gpu = m_flow_gpu / ( cp.abs( m_flow_gpu ) + l1Smooth )
            L1Grad_m_flow_gpu = L1Grad_m_flow_gpu[np.array( [1,0] ) ,:,:,:] - L1Grad_m_flow_gpu
            G0_m_gpu = Lambda2 * L1Grad_m_flow_gpu
            m_flow_gpu = None
            L1Grad_m_flow_gpu = None

            # Do L2-norm part
            L2Grad_degrid = cp.asnumpy(L2Grad_degrid_gpu)
            L2Grad_degrid_gpu = None
            L2Grad_gpu = cp.zeros(dims_L2Grad, dtype=np.complex64)
            for flow_idx in range(extra_dim2):
                # remove pinned memory as those block sizes will not be used again.
                pinned_mempool.free_all_blocks()
                mempool.free_all_blocks()
                self.log.debug("GPU memory info - iteration: " + str(i+1) + " - used: " + str(mempool.used_bytes()) + ", total: " + str(mempool.total_bytes()) + " pinned :" + str(pinned_mempool.n_free_blocks()) + " L2Grad in flow loop with flow_idx = " + str(flow_idx) )
                L2Grad_per_flow_cpu = kaiser2D.grid2D(L2Grad_degrid[:,flow_idx:flow_idx+1,...], coords[flow_idx:flow_idx+1,...], weights[flow_idx:flow_idx+1,...], kernel, out_dims_grid_per_flow, number_threads=number_threads)
                self.log.debug("GPU memory degub: L2Grad_per_flow_cpu.shape = " + str(L2Grad_per_flow_cpu.shape) + ", and dtype = " + str(L2Grad_per_flow_cpu.dtype))
                L2Grad_per_flow_gpu = cp.asarray(L2Grad_per_flow_cpu, dtype=np.complex64)

                for coil_idx in range(nr_coils):
                    L2Grad_per_flow_gpu[coil_idx,...] = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(L2Grad_per_flow_gpu[coil_idx,...], axes=(-1,-2)), axes=(-1,-2)), axes=(-1,-2))

                L2Grad_per_flow_gpu *= roll_gpu
                for phase_idx in range(extra_dim1):
                    L2Grad_per_flow_gpu[:,0,phase_idx,:,:] = csm_conj_gpu[:,0,0,:,:] * L2Grad_per_flow_gpu[:,0,phase_idx,:,:] # broadcast multiply to remove coil phase

                L2Grad_gpu[flow_idx:flow_idx+1,...] = 2. * L2Grad_per_flow_gpu.sum(axis=0) # assume the coil dim is the first
                L2Grad_per_flow_gpu = None
            
            # L1-norm cardiac part  performed on channel combined data
            L1Grad_card_gpu = w_card_gpu / ( cp.abs( w_card_gpu ) + l1Smooth )
            L1Grad_card_gpu = L1Grad_card_gpu[:, np.array( [-1] + list(range(L1Grad_card_gpu.shape[1]-1)) ) ,:,:] - L1Grad_card_gpu

            G1_gpu = L2Grad_gpu + ( Lambda * L1Grad_card_gpu )
            L2Grad_gpu = None
            w_card_gpu = None
            L1Grad_card_gpu = None

            if ((i)%reset_dx_iter == 0):
                bk = 0.
                self.log.info("\tNon-Lin CG - iteration = " + str(i+1) + ", bk = "+str(cp.asnumpy( bk )) )
                dx_gpu =  -G1_gpu.copy()
                t0 = 1.
            else:
                bk = cp.sum( cp.square (cp.abs( G1_gpu ))) / ( cp.sum( cp.square (cp.abs( G0_gpu ))) + eps)
                self.log.info("\tNon-Lin CG - iteration = " + str(i+1) + ", bk = "+str(cp.asnumpy( bk )) )
                dx_gpu =  -G1_gpu + bk * dx_gpu
            G0_gpu  = G1_gpu.copy()
            G1_gpu = None
            # remove pinned memory as those block sizes will not be used again.
            pinned_mempool.free_all_blocks()
            mempool.free_all_blocks()

            # backtracking line-search
            f0 = f1
            self.log.debug("Debugging non-lin CG iter: " + str(i+1) + ", prior to line search, f0 = " + str( f0)  + ", t = 0, L2Obj = " + str( L2Obj ) + ", L1Obj_card * Lambda = " + str( Lambda * L1Obj_card ) + ", L1Obj_flow * Lambda2 = " + str( Lambda2 * L1Obj_flow ))
            
            line_search_L2 = L2Obj.copy()
            line_search_L1_card = L1Obj_card.copy()
            line_search_L1_flow = L1Obj_flow.copy()
            
            t = t0
            x_t_gpu = (( cp.abs( x_0_gpu ) - t * G0_m_gpu ) * cp.exp( 1j * cp.angle(x_0_gpu) )) + t * dx_gpu

            # L2-norm part  performed on each channel
            L2Grad_degrid_gpu = cp.zeros(out_dims_degrid, dtype=np.complex64)
            self.log.debug("GPU memory info - iteration: " + str(i+1) + " - used: " + str(mempool.used_bytes()) + ", total: " + str(mempool.total_bytes()) + " pinned :" + str(pinned_mempool.n_free_blocks()) + ", before creating L2Grad_per_flow_gpu." )
            for flow_idx in range(extra_dim2):
                
                self.log.debug("GPU memory info - iteration: " + str(i+1) + " - used: " + str(mempool.used_bytes()) + ", total: " + str(mempool.total_bytes()) + " pinned :" + str(pinned_mempool.n_free_blocks()) + " L2-norm in flow loop with flow_idx = " + str(flow_idx) )
                L2Grad_per_flow_gpu = cp.zeros(dims_L2Grad_per_flow, dtype=np.complex64)
                for phase_idx in range(extra_dim1):
                    L2Grad_per_flow_gpu[:,0,phase_idx,...] = csm_gpu[:,0,0,...] * x_t_gpu[flow_idx,phase_idx,...] # add coil phase
                L2Grad_per_flow_gpu *= roll_gpu # pre-rolloff for degrid convolution
                # did not help: cp.fft.config.enable_nd_planning = False  # try reducing the memory needs on GPU possibly at cost of performance
                # want to do this but runs out of memory L2Grad = cp.asnumpy(cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(L2Grad_gpu), axes=(-1,-2))))
                for coil_idx in range(nr_coils):
                    L2Grad_per_flow_gpu[coil_idx,...] = cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(L2Grad_per_flow_gpu[coil_idx,...], axes=(-1,-2)), axes=(-1,-2)), axes=(-1,-2))
                L2Grad_degrid_gpu[:,flow_idx:flow_idx+1,...] = cp.asarray( kaiser2D.degrid2D(cp.asnumpy(L2Grad_per_flow_gpu), coords[flow_idx:flow_idx+1,...], kernel, out_dims_degrid_per_flow, number_threads=number_threads, oversampling_ratio = oversampling_ratio) )
                L2Grad_per_flow_gpu = None
            self.log.debug("max degridded image = "+str(cp.abs(L2Grad_degrid_gpu).max())+" vs max of data = "+str(cp.abs(data_gpu).max()) )
            L2Grad_degrid_gpu -= data_gpu
            L2Obj = cp.asnumpy(cp.sum( cp.square( (cp.abs(L2Grad_degrid_gpu))))).astype(np.float32)

            # L1-norm cardiac part  performed on channel combined data
            w_card_gpu = x_t_gpu.copy()
            w_card_gpu = w_card_gpu[:, 1 + np.array( list(range(w_card_gpu.shape[1]-1)) + [-1] ) ,:,:] - w_card_gpu
            L1Obj_card = cp.asnumpy( cp.sum( cp.abs( w_card_gpu ) ) )

            # L1-norm flow part    only compare magnitude between flow encodes and keep phase (velocity) unchanged
            m_flow_gpu = cp.abs(x_t_gpu)
            m_flow_gpu = m_flow_gpu[np.array( [1,0] ) ,:,:,:] - m_flow_gpu
            L1Obj_flow = cp.asnumpy( cp.sum( cp.abs( m_flow_gpu ) ) )

            f1 = (L2Obj / initial_L2) + ( L1Obj_card / initial_L1_card ) + ( L1Obj_flow / initial_L1_flow )
            self.log.info("Debugging non-lin CG iter: " + str(i+1) + ", prior to line search iter, t = " + str( t )  + ", L2 reduction: " + str(line_search_L2 / L2Obj) + ", cardiac L1 reduction: " + str(line_search_L1_card / L1Obj_card) + ", flow L1 reduction: " + str(line_search_L1_flow / L1Obj_flow))
            
            lsiter = 0;
            self.log.debug("Debugging non-lin CG f1 = "+str(f1)+" > f0 * (1 - (alpha * t)) = f0 * "+str(1 - (alpha * t))+" = "+str(f0 * (1 - (alpha * t))))
            while ( (f1 > f0 * (1 - (alpha * t))) and (lsiter<maxlsiter) ):
                lsiter = lsiter + 1
                t = t * beta
                
                # updating these, delete them first to create GPU memory
                L2Grad_degrid_gpu = None
                w_card_gpu = None
                m_flow_gpu = None
                x_t_gpu = (( cp.abs( x_0_gpu ) - t * G0_m_gpu ) * cp.exp( 1j * cp.angle(x_0_gpu) )) + t * dx_gpu
                L2Grad_degrid_gpu = cp.zeros(out_dims_degrid, dtype=np.complex64)
                for flow_idx in range(extra_dim2):
                    # remove pinned memory as those block sizes will not be used again.
                    pinned_mempool.free_all_blocks()
                    mempool.free_all_blocks()
                    self.log.debug("GPU memory info - iteration: " + str(i+1) + " - used: " + str(mempool.used_bytes()) + ", total: " + str(mempool.total_bytes()) + " pinned :" + str(pinned_mempool.n_free_blocks()) + " L2-norm in while looo in flow loop with flow_idx = " + str(flow_idx) )
                    L2Grad_per_flow_gpu = cp.zeros(dims_L2Grad_per_flow, dtype=np.complex64)
                    for phase_idx in range(extra_dim1):
                        L2Grad_per_flow_gpu[:,0,phase_idx,...] = csm_gpu[:,0,0,...] * x_t_gpu[flow_idx,phase_idx,...] # add coil phase
                    L2Grad_per_flow_gpu *= roll_gpu # pre-rolloff for degrid convolution
                    # did not help: cp.fft.config.enable_nd_planning = False  # try reducing the memory needs on GPU possibly at cost of performance
                    # want to do this but runs out of memory L2Grad = cp.asnumpy(cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(L2Grad_gpu), axes=(-1,-2))))
                    for coil_idx in range(nr_coils):
                        L2Grad_per_flow_gpu[coil_idx,...] = cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(L2Grad_per_flow_gpu[coil_idx,...], axes=(-1,-2)), axes=(-1,-2)), axes=(-1,-2))
                    L2Grad_degrid_gpu[:,flow_idx:flow_idx+1,...] = cp.asarray( kaiser2D.degrid2D(cp.asnumpy(L2Grad_per_flow_gpu), coords[flow_idx:flow_idx+1,...], kernel, out_dims_degrid_per_flow, number_threads=number_threads, oversampling_ratio = oversampling_ratio) )
                    L2Grad_per_flow_gpu = None
                self.log.debug("max degridded image = "+str(cp.abs(L2Grad_degrid_gpu).max())+" vs max of data = "+str(cp.abs(data_gpu).max()) )
                self.log.debug("compare shapes of L2Grad_degrid_gpu vs data: "+str(L2Grad_degrid_gpu.shape)+" and "+str(data_gpu.shape))
                L2Grad_degrid_gpu -= data_gpu
                L2Obj = cp.asnumpy(cp.sum( cp.square( (cp.abs(L2Grad_degrid_gpu))))).astype(np.float32)

                # L1-norm cardiac part  performed on channel combined data
                w_card_gpu = x_t_gpu.copy()
                w_card_gpu = w_card_gpu[:, 1 + np.array( list(range(w_card_gpu.shape[1]-1)) + [-1] ) ,:,:] - w_card_gpu
                L1Obj_card = cp.asnumpy( cp.sum( cp.abs( w_card_gpu ) ) )

                # L1-norm flow part  only compare magnitude between flow encodes and keep phase (velocity) unchanged
                m_flow_gpu = cp.abs(x_t_gpu)
                m_flow_gpu = m_flow_gpu[np.array( [1,0] ) ,:,:,:] - m_flow_gpu
                L1Obj_flow = cp.asnumpy( cp.sum( cp.abs( m_flow_gpu ) ) )

                f1 = (L2Obj / initial_L2) + ( L1Obj_card / initial_L1_card ) + ( L1Obj_flow / initial_L1_flow )
                self.log.info("Debugging non-lin CG iter: " + str(i+1) + ", line search iter: " + str( lsiter ) + ", t = " + str( t )  + ", L2 reduction: " + str(line_search_L2 / L2Obj) + ", cardiac L1 reduction: " + str(line_search_L1_card / L1Obj_card) + ", flow L1 reduction: " + str(line_search_L1_flow / L1Obj_flow))
                self.log.debug("Debugging non-lin CG f1 = "+str(f1)+" > f0 * (1 - (alpha * t)) = f0 * "+str(1 - (alpha * t))+" = "+str(f0 * (1 - (alpha * t))))

            if lsiter == maxlsiter:
                self.log.warn("Warning - line search reached " + str(maxlsiter) + " iterations. Continue anyway..")
                #return 1;

            # control the number of line searches by adapting the initial step search
            if lsiter >= 2:
                t0 *= beta
            if lsiter >= 4:
                t0 *= beta
            if lsiter >= 6:
                t0 *= beta
            if lsiter >= 8:
                t0 *= beta
            if lsiter < 1:
                t0 /= beta

            x_0_gpu = x_t_gpu.copy()
            x_t_gpu = None

            self.log.node("\tNon-Lin CG - iteration = " + str(i+1) + ", cost = " + str(f1) + ", cost reduction: " + str(initial_f0 / f1) + ", L2 reduction: " + str(initial_L2 / L2Obj) + ", cardiac L1 reduction: " + str(initial_L1_card / L1Obj_card)  + ", respiratory L1 reduction: " + str(initial_L1_flow / L1Obj_flow))

            cost_time_array[i,0] = time.time() - time_0
            cost_time_array[i,1] = initial_f0 / f1

            current_iteration = cp.asnumpy(x_0_gpu)
            current_iteration.shape = iterations_shape
            if store_iterations:
                x_iterations[i+1,:,:,:,:] = current_iteration[..., mtx_min:mtx_max, mtx_min:mtx_max]

            self.log.debug("GPU memory info - iteration = " + str(i+1) + " - used: " + str(mempool.used_bytes()) + ", total: " + str(mempool.total_bytes()) + " pinned :" + str(pinned_mempool.n_free_blocks()) + ", end of iteration loop." )


        # return the final image     
        self.setData('out', np.squeeze(current_iteration[..., mtx_min:mtx_max, mtx_min:mtx_max]))
        self.setData('cost vs. time', cost_time_array)
        if store_iterations:
            self.setData('x iterations', np.squeeze(x_iterations))

        return 0

    def execType(self):
        #return gpi.GPI_THREAD
        return gpi.GPI_PROCESS
