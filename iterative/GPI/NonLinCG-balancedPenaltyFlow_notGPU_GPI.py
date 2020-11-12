# Copyright (c)
# Author: Mike Schär
# Date: 2017 July 28

import numpy as np
import gpi

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
        step: execute an additional iteration (will add to 'iterations')
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
        self.log.debug("validate NonLinCG-balancedPenaltyFlow")
        
        # check size of data vs. coords
        self.log.debug("validate NonLinCG-balancedPenaltyFlow - check size of data vs. coords")
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
        
        self.log.debug("validate NonLinCG-balancedPenaltyFlow - check csm")
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
        out_dims_degrid = [nr_coils, extra_dim2, extra_dim1, nr_arms, nr_points]
        out_dims_fft = [nr_coils, extra_dim2, extra_dim1, mtx, mtx]
        iterations_shape = [extra_dim2, extra_dim1, mtx, mtx]
        
        # coords dimensions: (like data, coords should have 5 dimensions)
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
            gridded_kspace = kaiser2D.grid2D(data, coords, weights, kernel, out_dims_grid, number_threads=number_threads)
            
            # FFT
            image_domain = kaiser2D.fft2D(gridded_kspace, dir=0, out_dims_fft=out_dims_fft)
            # rolloff
            image_domain *= roll
            
            if csm is None:
                self.log.info("Generating autocalibrated B1 maps... use flow encode 0 only")
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
                extra_dim2_csm = 1
                extra_dim1_csm = 1
                
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
                if extra_dim1 > 1:
                    csm = np.zeros([nr_coils, extra_dim2, extra_dim1, mtx, mtx], dtype=image_domain_csm.dtype)
                    for extra_dim1_counter in range(extra_dim1):
                        csm[:,:,extra_dim1_counter,:,:]=image_domain_csm[:,:,0,:,:]
                else:
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
    
                    
        else: # this is the normal path (not single iteration step)
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

        # keep a conjugate csm set on hand
        csm_conj = np.conj(csm)
        
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
        eps = np.finfo( image_domain.dtype ).eps
        self.log.debug("eps = " + str(eps) + ", and 1./l1Smooth = " + str(1./l1Smooth))
        reset_dx_iter = 5
        # for debugging store cost and time
        # cost_time_array[ iterations, 2 ] with time and cost
        cost_time_array = np.zeros([ iterations, 2 ], dtype=np.float32)

        # calculate initial conditions
        x_0 = csm_conj * image_domain # broadcast multiply to remove coil phase
        x_0 = x_0.sum(axis=0) # assume the coil dim is the first
        Lambda = lamb * np.max( np.abs( x_0 ) )
        Lambda2 = lamb2 * np.max( np.abs( x_0 ) )
        self.log.info("Lambda (cardiac) = "+str(Lambda) + " and Lambda2 (flow) = "+str(Lambda2))

        current_iteration = x_0.copy()
        current_iteration.shape = iterations_shape
        if store_iterations:
            x_iterations[0,:,:,:,:] = current_iteration[...,mtx_min:mtx_max,mtx_min:mtx_max]

        # during iterations, the objective is re-calculated in the process
        # when line-search is finished, the gradient is calculated for the next iteation. The gradient calculation can use some values from the last objective calculation.
        # Therefore, here, first the objective is calculated, and the gradient will be calculated in the first step within the iteration loop.

        # L2-norm part  performed on each channel
        L2Grad = csm * x_0 # add coil phase
        L2Grad *= roll # pre-rolloff for degrid convolution
        L2Grad = kaiser2D.fft2D(L2Grad, dir=1)
        L2Grad = kaiser2D.degrid2D(L2Grad, coords, kernel, out_dims_degrid, number_threads=number_threads, oversampling_ratio = oversampling_ratio)
        self.log.debug("max degridded image = "+str(np.abs(L2Grad).max())+" vs max of data = "+str(np.abs(data).max()) )
        self.log.debug("compare shapes of L2Grad vs data: "+str(L2Grad.shape)+" and "+str(data.shape))
        L2Grad -= data
        L2Obj = np.sum( np.square( (np.abs(L2Grad)))).astype(np.float32)

        # L1-norm part  performed on channel combined data
        w_card = x_0.copy()
        w_card = w_card[:, 1 + np.array( list(range(w_card.shape[1]-1)) + [-1] ) ,:,:] - w_card
        L1Obj_card = np.sum( np.abs( w_card ) )

        # L1-norm flow part    only compare magnitude between flow encodes and keep phase (velocity) unchanged
        m_flow = np.abs(x_0)
        m_flow = m_flow[np.array( [1,0] ) ,:,:,:] - m_flow
        L1Obj_flow = np.sum( np.abs( m_flow ) )

        # before balancing penalty f0 = L2Obj + ( Lambda * L1Obj )
        # seems the scaling between objective and gradient are very different, which leads to the effect that when using the same lambda for scaling, either component has no effect in either step.
        # Therefore, here we weight each component as 1 for the objective, and use the lambda scaling factor to weight the relative contributions to the gradient.
        f0 = 3.0
        f1 = 3.0
        self.log.info("Debugging non-lin CG iter:  1, prior to line search, f0 = " + str( f0)  + ", t = 0, L2Obj = " + str( L2Obj ) + ", L1Obj_card * Lambda = " + str( Lambda * L1Obj_card ) + ", L1Obj_flow * Lambda2 = " + str( Lambda2 * L1Obj_flow ))

        # initial objectives
        initial_L2 = L2Obj.copy()
        initial_L1_card = L1Obj_card.copy()
        initial_L1_flow = L1Obj_flow.copy()
        initial_f0 = f0

        # conjugate gradient calculation with backtracking line search
        for i in range(iterations):
            # determine the gradient
            
            # Do L2-norm part
            L2Grad = kaiser2D.grid2D(L2Grad, coords, weights, kernel, out_dims_grid, number_threads=number_threads)
            L2Grad = kaiser2D.fft2D(L2Grad, dir=0)
            L2Grad *= roll
            L2Grad = csm_conj * L2Grad # broadcast multiply to remove coil phase
            L2Grad = 2. * L2Grad.sum(axis=0) # assume the coil dim is the first
            
            # L1-norm cardiac part  performed on channel combined data
            L1Grad_card = w_card / ( np.abs( w_card ) + l1Smooth )
            L1Grad_card = L1Grad_card[:, np.array( [-1] + list(range(L1Grad_card.shape[1]-1)) ) ,:,:] - L1Grad_card
            
            # L1-norm flow part   only compare magnitude between flow encodes and keep phase (velocity) unchanged
            L1Grad_m_flow = m_flow / ( np.abs( m_flow ) + l1Smooth )
            L1Grad_m_flow = L1Grad_m_flow[np.array( [1,0] ) ,:,:,:] - L1Grad_m_flow
            
            G1 = L2Grad + ( Lambda * L1Grad_card ) #+ ( Lambda2 * L1Grad_flow )
            G0_m = Lambda2 * L1Grad_m_flow
            
            if ((i)%reset_dx_iter == 0):
                bk = 0.
                self.log.info("\tNon-Lin CG - iteration = " + str(i+1) + ", bk = "+str( bk ) )
                dx = -G1
                t0 = 1.
            else:
                #bk = np.real( np.dot(np.conj( G1.flatten() ), G1.flatten() ) / ( np.dot(np.conj( G0.flatten() ), G0.flatten() ) + eps) )
                bk = np.sum( np.square(np.abs( G1 ))) / ( np.sum(np.square(np.abs( G0 ))) + eps)
                self.log.info("\tNon-Lin CG - iteration = " + str(i+1) + ", bk = "+str( bk ) )
                dx =  - G1 + bk * dx;
            
            G0  = G1.copy()
            
            
            # backtracking line search
            f0 = f1
            self.log.debug("Debugging non-lin CG iter: " + str(i+1) + ", prior to line search, f0 = " + str( f0)  + ", t = 0, L2Obj = " + str( L2Obj ) + ", L1Obj_card * Lambda = " + str( Lambda * L1Obj_card ) + ", L1Obj_flow * Lambda2 = " + str( Lambda2 * L1Obj_flow ))

            line_search_L2 = L2Obj.copy()
            line_search_L1_card = L1Obj_card.copy()
            line_search_L1_flow = L1Obj_flow.copy()
            
            t = t0
            x_t = (( np.abs( x_0 ) - t * G0_m ) * np.exp( 1j * np.angle(x_0) )) + t * dx

            # L2-norm part  performed on each channel
            L2Grad = x_t.copy()
            L2Grad = csm * L2Grad # add coil phase
            L2Grad *= roll # pre-rolloff for degrid convolution
            L2Grad = kaiser2D.fft2D(L2Grad, dir=1)
            L2Grad = kaiser2D.degrid2D(L2Grad, coords, kernel, out_dims_degrid, number_threads=number_threads, oversampling_ratio = oversampling_ratio)
            L2Grad -= data
            #L2Obj = np.real( np.dot(np.conj(L2Grad.flatten()), L2Grad.flatten()) )
            L2Obj = np.sum( np.square( (np.abs(L2Grad)))).astype(np.float32)

            # L1-norm cardiac part  performed on channel combined data
            w_card = x_t.copy()
            w_card = w_card[:, 1 + np.array( list(range(w_card.shape[1]-1)) + [-1] ) ,:,:] - w_card
            L1Obj_card = np.sum( np.abs( w_card ) )

            # L1-norm flow part    only compare magnitude between flow encodes and keep phase (velocity) unchanged
            m_flow = np.abs(x_t)
            m_flow = m_flow[np.array( [1,0] ) ,:,:,:] - m_flow
            L1Obj_flow = np.sum( np.abs( m_flow ) )

            f1 = (L2Obj / initial_L2) + ( L1Obj_card / initial_L1_card ) + ( L1Obj_flow / initial_L1_flow )
            self.log.info("Debugging non-lin CG iter: " + str(i+1) + ", prior to line search, t = " + str( t )  + ", L2 reduction: " + str(line_search_L2 / L2Obj) + ", cardiac L1 reduction: " + str(line_search_L1_card / L1Obj_card) + ", flow L1 reduction: " + str(line_search_L1_flow / L1Obj_flow))
            
            lsiter = 0;
            self.log.debug("Debugging non-lin CG f1 = "+str(f1)+" > f0 * (1 - (alpha * t)) = f0 * "+str(1 - (alpha * t))+" = "+str(f0 * (1 - (alpha * t))))
            while ( (f1 > f0 * (1 - (alpha * t))) and (lsiter<maxlsiter) ):
                lsiter = lsiter + 1
                t = t * beta
                x_t = (( np.abs( x_0 ) - t * G0_m ) * np.exp( 1j * np.angle(x_0) )) + t * dx
                L2Grad = x_t.copy()
                L2Grad = csm * L2Grad # add coil phase
                L2Grad *= roll # pre-rolloff for degrid convolution
                L2Grad = kaiser2D.fft2D(L2Grad, dir=1)
                L2Grad = kaiser2D.degrid2D(L2Grad, coords, kernel, out_dims_degrid, number_threads=number_threads, oversampling_ratio = oversampling_ratio)
                L2Grad -= data
                L2Obj = np.sum( np.square( (np.abs(L2Grad)))).astype(np.float32)

                # L1-norm cardiac part  performed on channel combined data
                w_card = x_t.copy()
                w_card = w_card[:, 1 + np.array( list(range(w_card.shape[1]-1)) + [-1] ) ,:,:] - w_card
                L1Obj_card = np.sum( np.abs( w_card ) )
                
                # L1-norm flow part    only compare magnitude between flow encodes and keep phase (velocity) unchanged
                m_flow = np.abs(x_t)
                m_flow = m_flow[np.array( [1,0] ) ,:,:,:] - m_flow
                L1Obj_flow = np.sum( np.abs( m_flow ) )
                
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

            x_0 = x_t.copy()

            self.log.node("\tNon-Lin CG - iteration = " + str(i+1) + ", cost = " + str(f1) + ", cost reduction: " + str(initial_f0 / f1) + ", L2 reduction: " + str(initial_L2 / L2Obj) + ", cardiac L1 reduction: " + str(initial_L1_card / L1Obj_card)  + ", flow L1 reduction: " + str(initial_L1_flow / L1Obj_flow))
            cost_time_array[i,0] = time.time() - time_0
            cost_time_array[i,1] = initial_f0 / f1

            current_iteration = x_0.copy()
            current_iteration.shape = iterations_shape
            if store_iterations:
                x_iterations[i+1,:,:,:,:] = current_iteration[..., mtx_min:mtx_max, mtx_min:mtx_max]

        # return the final image     
        self.setData('out', np.squeeze(current_iteration[..., mtx_min:mtx_max, mtx_min:mtx_max]))
        self.setData('cost vs. time', cost_time_array)
        if store_iterations:
            self.setData('x iterations', np.squeeze(x_iterations))

        return 0

    def execType(self):
        #return gpi.GPI_THREAD
        return gpi.GPI_PROCESS
