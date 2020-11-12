# copyright 2020 Michael Schar

import gpi
import numpy as np
from gpi import QtGui

class ExternalNode(gpi.NodeAPI):
    """ Array compression for virtual channels
        Reduce the number of channels to a user defined number based on the publication by Buehrer Martin et al, MRM 2007
    """

    def initUI(self):
        # Widgets
        self.addWidget('SpinBox', 'mtx', val=300, min=1)
        self.addWidget('ExclusivePushButtons','trajectory', buttons=['spiral or radial','Cartesian'], val=0)
        self.addWidget('DisplayBox', 'image', interp=True, ann_box='Pointer')
        self.addWidget('Slider', 'image ceiling',val=60)
        self.addWidget('Slider', 'crop left')
        self.addWidget('Slider', 'crop right')
        self.addWidget('Slider', 'crop top')
        self.addWidget('Slider', 'crop bottom')
        self.addWidget('PushButton', 'reset compute button each time', toggle=True, val=1, collapsed=True)
        self.addWidget('PushButton', 'compute', toggle=True)
        self.addWidget('SpinBox', 'virtual channels', immediate=True, val=12)
        self.addWidget('Slider', 'Autocalibration Width (%)', val=10, min=0, max=100)
        self.addWidget('Slider', 'Autocalibration Taper (%)', val=50, min=0, max=100)
        self.addWidget('Slider', 'Mask Floor (% of max mag)', val=5, min=0, max=100)
        self.addWidget('Slider', '# golden angle dynamics for csm', val=150, min=0, max=1000)
        self.addWidget('PushButton', 'Dynamic data - average all dynamics for csm', toggle=True, button_title='ON', val=1)
        self.addWidget('SpinBox','SDC Iterations',val=10, min=1)

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=[np.complex64, np.complex128], obligation=gpi.REQUIRED)
        self.addInPort('noise', 'NPYarray', dtype=[np.complex64, np.complex128], obligation=gpi.REQUIRED)
        self.addInPort('coords', 'NPYarray', dtype=[np.float32, np.float64], obligation=gpi.OPTIONAL)
        self.addInPort('sensitivity map', 'NPYarray', dtype=[np.complex64, np.complex128], obligation=gpi.OPTIONAL)
        self.addInPort('params_in', 'DICT', obligation = gpi.OPTIONAL)

        self.addOutPort('compressed data', 'NPYarray')
        self.addOutPort('A', 'NPYarray')
        self.addOutPort('noise covariance', 'NPYarray')
        self.addOutPort('masked and normalized sense map', 'NPYarray')
        self.addOutPort('sum of square image', 'NPYarray')
        self.addOutPort('debug', 'NPYarray')

    
    def validate(self):
        data=self.getData('data')
        coords=self.getData('coords')
        param = self.getData('params_in')
        
        if coords is not None:
            if coords.shape[-1] != 2:
                self.log.warn("Currently only for 2D data")
                return 1

        if param is not None:
            if ( ('spFOVXY' in param) and ('spRESXY' in param) ):
                mtx_xy = 1.25*float(param['spFOVXY'][0])/float(param['spRESXY'][0])
                self.setAttr('mtx', quietval = mtx_xy)
            if 'spDYN_GOLDANGLE_ON' in param:
                if int(param['spDYN_GOLDANGLE_ON'][0]) == 1:
                    self.setAttr('# golden angle dynamics for csm', visible=True)
                    self.setAttr('# golden angle dynamics for csm', max=coords.shape[-3])
                else:
                    self.setAttr('# golden angle dynamics for csm', visible=False)
        else:
            self.setAttr('# golden angle dynamics for csm', visible=True)
            self.setAttr('# golden angle dynamics for csm', max=coords.shape[-3])
        
        self.log.debug("validate VirtualChannels - check csm")
        csm = self.getData('sensitivity map')
        if csm is None:
            self.setAttr('Autocalibration Width (%)', visible=True)
            self.setAttr('Autocalibration Taper (%)', visible=True)
            self.setAttr('Mask Floor (% of max mag)', visible=True)
            self.setAttr('SDC Iterations', visible=True)
            if data.ndim > 3:
                self.setAttr('Dynamic data - average all dynamics for csm', visible=True)
            else:
                self.setAttr('Dynamic data - average all dynamics for csm', visible=False)
            UI_width = self.getVal('Autocalibration Width (%)')
            csm_mtx = np.int(0.01 * UI_width * self.getVal('mtx'))
        else:
            self.setAttr('Autocalibration Width (%)', visible=False)
            self.setAttr('Autocalibration Taper (%)', visible=False)
            self.setAttr('Mask Floor (% of max mag)', visible=False)
            self.setAttr('Dynamic data - average all dynamics for csm', visible=False)
            self.setAttr('SDC Iterations', visible=False)
            csm_mtx = csm.shape[-1]
        
        if ( (len(self.portEvents() ) > 0) or ('Autocalibration Width (%)' in self.widgetEvents()) ):
            self.setAttr('crop left', max=csm_mtx, min=1)
            self.setAttr('crop right', max=csm_mtx, min=1)
            self.setAttr('crop top', max=csm_mtx, min=1)
            self.setAttr('crop bottom', max=csm_mtx, min=1)
        if 'crop left' in self.widgetEvents():
            value_below = self.getVal('crop left')
            value_above = self.getVal('crop right')
            if value_below == csm_mtx:
                self.setAttr('crop left', quietval=csm_mtx-1)
            if value_above <= value_below:
                self.setAttr('crop right', quietval=value_below+1)
        if 'crop right' in self.widgetEvents():
            value_below = self.getVal('crop left')
            value_above = self.getVal('crop right')
            if value_above == 1:
                self.setAttr('crop right', quietval=2)
            if value_above <= value_below:
                self.setAttr('crop left', quietval=value_above-1)
        if 'crop top' in self.widgetEvents():
            value_below = self.getVal('crop top')
            value_above = self.getVal('crop bottom')
            if value_below == csm_mtx:
                self.setAttr('crop top', quietval=csm_mtx-1)
            if value_above <= value_below:
                self.setAttr('crop bottom', quietval=value_below+1)
        if 'crop bottom' in self.widgetEvents():
            value_below = self.getVal('crop top')
            value_above = self.getVal('crop bottom')
            if value_above == 1:
                self.setAttr('crop bottom', quietval=2)
            if value_above <= value_below:
                self.setAttr('crop top', quietval=value_above-1)

    def compute(self):
        import numpy as np
        from scipy import linalg

        self.log.node("Virtual Channels node running compute()")
        twoD_or_threeD = 2

        # GETTING WIDGET INFO
        mtx_xy = self.getVal('mtx')
        trajectory = self.getVal('trajectory')
        image_ceiling = self.getVal('image ceiling')
        crop_left = self.getVal('crop left')
        crop_right = self.getVal('crop right')
        crop_top = self.getVal('crop top')
        crop_bottom = self.getVal('crop bottom')
        reset_compute_button = self.getVal('reset compute button each time')
        compute = self.getVal('compute')
        # number of virtual channels m
        m = self.getVal('virtual channels')
        numiter = self.getVal('SDC Iterations')
        
        # GETTING PORT INFO
        data = self.getData('data').astype(np.complex64, copy=False)
        noise = self.getData('noise')
        sensitivity_map_uncropped = self.getData('sensitivity map')
        param = self.getData('params_in')
        
        # set dimensions
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
        elif data.ndim > 5:
            self.log.warn("Not implemented yet")
        
        print("data shape: " + str(data.shape))
        

        if sensitivity_map_uncropped is None:
            # if cropping or image scaling sliders were changed then use the previously stored csm instead of calcluating a new one
            has_csm_been_calculated = self.getData('sum of square image')
            if ( (has_csm_been_calculated is not None) and
                ( ('crop left' in self.widgetEvents()) or
                ('crop right' in self.widgetEvents()) or
                ('crop top' in self.widgetEvents()) or
                ('crop bottom' in self.widgetEvents()) or
                ('image ceiling' in self.widgetEvents()) ) ):
                    csm = self.getData('masked and normalized sense map')
                    image = self.getData('sum of square image').copy()
                    if ( (csm is None) or (image is None) ):
                        self.log.warn("This should not happen.")
                        return 1
                    csm_mtx = csm.shape[-1]
            else:
                # calculate auto-calibration B1 maps
                import triggeredGASSP.gridding.Kaiser2D_utils as kaiser2D
                UI_mask_floor = self.getVal('Mask Floor (% of max mag)')
                if (trajectory == 0):
                    # spiral or radial
                    coords = self.getData('coords').astype(np.float32, copy=False)
                    if (coords is None):
                        self.log.warn("Either a sensitiviy map or coords to calculate one is required")
                        return 1
                    
                    # parameters from UI
                    UI_width = self.getVal('Autocalibration Width (%)')
                    UI_taper = self.getVal('Autocalibration Taper (%)')
                    #UI_mask_floor = self.getVal('Mask Floor (% of max mag)')
                    UI_average_csm = self.getVal('Dynamic data - average all dynamics for csm')
                    csm_mtx = np.int(0.01 * UI_width * mtx_xy)
                    
                    if coords.shape[-3]>400:
                        is_GoldenAngle_data = True
                    else:
                        is_GoldenAngle_data = False
                    if param is not None:
                        if 'spDYN_GOLDANGLE_ON' in param:
                            if int(param['spDYN_GOLDANGLE_ON'][0]) == 1:
                                is_GoldenAngle_data = True
                            else:
                                is_GoldenAngle_data = False
                    print("is GoldenAngle: " + str(is_GoldenAngle_data))
                    if is_GoldenAngle_data:
                        nr_arms_cms = self.getVal('# golden angle dynamics for csm')
                    else:
                        nr_arms_cms = nr_arms
                    self.log.debug("nr_arms_cms: " + str(nr_arms_cms))
                    csm_data = data[...,0:nr_arms_cms,:]

                    # oversampling: Oversample at the beginning and crop at the end
                    oversampling_ratio = 2. #1.375
                    mtx = np.int(csm_mtx * oversampling_ratio)
                    if mtx%2:
                        mtx+=1
                    if oversampling_ratio > 1:
                        mtx_min = np.int((mtx-csm_mtx)/2)
                        mtx_max = mtx_min + csm_mtx
                    else:
                        mtx_min = 0
                        mtx_max = mtx

                    # average dynamics or cardiac phases for csm
                    if ( (extra_dim1 > 1) and UI_average_csm ):
                        csm_data = np.sum(csm_data, axis=2)
                        csm_data = np.sum(csm_data, axis=1)  # Modified by Dan Zhu, simply add extra_dim2 as well
                        extra_dim1_csm = 1
                        #csm_data.shape = [nr_coils, extra_dim2, extra_dim1_csm, nr_arms_cms, nr_points]
                        extra_dim2_csm = 1 # Modified by Dan Zhu, simply add extra_dim2 as well
                        csm_data.shape = [nr_coils, extra_dim2_csm, extra_dim1_csm, nr_arms_cms, nr_points]
                        # Modified by Dan Zhu, simply add extra_dim2 as well

                    else:
                        extra_dim1_csm = extra_dim1
                        extra_dim2_csm = extra_dim2
                    
                    self.log.debug("csm_data shape: " + str(csm_data.shape))
                    self.log.debug("coords shape: " + str(coords.shape))

                    # coords dimensions: (add 1 dimension as they could have another dimension for golden angle dynamics
                    if coords.ndim == 3:
                        coords.shape = [1,nr_arms,nr_points,twoD_or_threeD]
                    
                    # create low resolution csm
                    # cropping the data will make gridding and FFT much faster
                    magnitude_one_interleave = np.zeros(nr_points)
                    for x in range(nr_points):
                        magnitude_one_interleave[x] = np.sqrt( coords[0,0,x,0]**2 + coords[0,0,x,1]**2)
                    self.log.debug(magnitude_one_interleave)
                    within_csm_width_radius = magnitude_one_interleave[:] < (0.01 * UI_width * 0.5) # for BNI spirals should be 0.45 instead of 0.5
                    nr_points_csm_width = within_csm_width_radius.sum()
                    
                    # now set the dimension lists
                    # Modified by Dan Zhu, simply add extra_dim2 as well
                    out_dims_grid = [nr_coils, extra_dim2_csm, extra_dim1_csm, mtx, nr_arms_cms, nr_points_csm_width]
                    out_dims_fft = [nr_coils, extra_dim2_csm, extra_dim1_csm, mtx, mtx]

                    csm_data = csm_data[...,0:nr_points_csm_width]
                    csm_coords = 1. / (0.01 * UI_width) * coords[...,0:nr_arms_cms,0:nr_points_csm_width,:]

                    self.log.debug("csm_data shape: " + str(csm_data.shape))
                    self.log.debug("csm_coords shape: " + str(csm_coords.shape))
                    
                    # generate SDC based on number of arms and nr of points being used for csm
                    import gpi_core.gridding.sdc as sd
                    #csm_weights = sd.twod_sdcsp(csm_coords.squeeze().astype(np.float64), numiter, 0.01 * UI_taper, mtx)
                    cmtxdim = np.array([mtx,mtx],dtype=np.int64)
                    wates = np.ones((nr_arms_cms * nr_points_csm_width), dtype=np.float64)
                    coords_for_sdc = csm_coords.astype(np.float64)
                    coords_for_sdc.shape = [nr_arms_cms * nr_points_csm_width, twoD_or_threeD]
                    csm_weights = sd.twod_sdc(coords_for_sdc, wates, cmtxdim, numiter, 0.01 * UI_taper )
                    csm_weights.shape = [1,nr_arms_cms,nr_points_csm_width]

                    # pre-calculate Kaiser-Bessel kernel
                    kernel_table_size = 800
                    kernel = kaiser2D.kaiserbessel_kernel( kernel_table_size, oversampling_ratio)
                    
                    # pre-calculate the rolloff for the spatial domain
                    roll = kaiser2D.rolloff2D(mtx, kernel)
                    # Grid
                    gridded_kspace = kaiser2D.grid2D(csm_data, csm_coords, csm_weights.astype(np.float32), kernel, out_dims_grid)
                    self.setData('debug', gridded_kspace)
                    # filter k-space - not needed anymore as SDC taper is used now.
                    ## win = kaiser2D.window2(gridded_kspace.shape[-2:], windowpct=UI_taper, widthpct=100)
                    ## gridded_kspace *= win
                    # FFT
                    image_domain = kaiser2D.fft2D(gridded_kspace, dir=0, out_dims_fft=out_dims_fft)
                    # rolloff
                    image_domain *= roll
                    # crop to original matrix size
                    csm = image_domain[...,mtx_min:mtx_max,mtx_min:mtx_max]
                else:
                    # Cartesian
                    csm_mtx = nr_arms
                    extra_dim2_csm = 1
                    extra_dim1_csm = 1
                    mtx_min = (nr_points - nr_arms)//2
                    mtx_max = mtx_min + csm_mtx
                    out_dims_fft = [nr_coils, extra_dim2_csm, extra_dim1_csm, nr_arms, nr_points]
                    csm = kaiser2D.fft2D(data, dir=0, out_dims_fft=out_dims_fft)
                    csm = csm[...,mtx_min:mtx_max]
                # normalize by rms (better would be to use a whole body coil image
                csm_rms = np.sqrt(np.sum(np.abs(csm)**2, axis=0))
                csm = csm / csm_rms
                # zero out points that are below mask threshold
                thresh = 0.01 * UI_mask_floor * csm_rms.max()
                csm *= csm_rms > thresh
                # for ROI selection use csm_rms, which still has some contrast
                image = csm_rms
                image.shape = [csm_mtx,csm_mtx]
                image_sos = image.copy()
                self.setData('sum of square image', image_sos)
        else:
            csm = sensitivity_map_uncropped
            csm_mtx = csm.shape[-1]
            # create sum-of-squares of sensitivity map to allow selection of ROI
            image = np.copy(csm)
            image = np.sqrt(np.sum(np.abs(image)**2, axis=0))
            image.shape = [csm_mtx,csm_mtx]
        self.setData('masked and normalized sense map', csm)

        # display sum-of-squares of sensitivity map to allow selection of ROI
        data_max = image.max()
        data_min = image.min()

        image[:, crop_left-1] = data_max
        image[:, crop_right-1] = data_max
        image[crop_top-1, :] = data_max
        image[crop_bottom-1, :] = data_max
        
        data_range = data_max - data_min
        new_max = data_range * 0.01 * image_ceiling + data_min
        dmask = np.ones(image.shape)
        image = np.minimum(image,new_max*dmask)
        if new_max > data_min:
            image = 255.*(image - data_min)/(new_max-data_min)
        red = green = blue = np.uint8(image)
        alpha = 255. * np.ones(blue.shape)
        h, w = red.shape[:2]
        image1 = np.zeros((h, w, 4), dtype=np.uint8)
        image1[:, :, 0] = red
        image1[:, :, 1] = green
        image1[:, :, 2] = blue
        image1[:, :, 3] = alpha

        format_ = QtGui.QImage.Format_RGB32
        
        image2 = QtGui.QImage(image1.data, w, h, format_)
        image2.ndarry = image1
        self.setAttr('image', val=image2)

        # crop sensitivity map
        csm.shape = [nr_coils, csm_mtx, csm_mtx]
        sensitivity_map = csm[:,crop_top-1:crop_bottom,crop_left-1:crop_right]
        #self.setData('debug', np.squeeze(sensitivity_map))

        # get sizes
        # number of channels n
        n = sensitivity_map.shape[-3]
        x_size = sensitivity_map.shape[-1]
        y_size = sensitivity_map.shape[-2]
        nr_pixels = x_size * y_size

        if compute:

            # noise covariance matrix Psi
            noise_cv_matrix = np.cov(noise)

            # Cholesky decomposition to determine T, where T Psi T_H = 1
            L = np.linalg.cholesky(noise_cv_matrix)
            T = np.linalg.inv(L)

            # decorrelated sensitivity map S_hat
            S_hat = np.zeros([nr_pixels, n], dtype=np.complex64)
            for x in range(x_size):
                for y in range(y_size):
                    index = y + x * y_size
                    S_hat[index, :] = np.dot(T, sensitivity_map[:,y,x])
                        
            self.log.debug("after S_hat")
            
            # P = sum of S_hat S_hat_pseudo_inverse over all pixels
            P = np.zeros([n,n], dtype=np.complex64)
            S_hat_matrix = np.zeros([n,1], dtype=np.complex64)
            for index in range(nr_pixels):
                # pseudo inverse of S_hat
                S_hat_matrix[:,0] = S_hat[index,:]
                S_hat_pinv = np.linalg.pinv(S_hat_matrix)
                P = P + np.dot(S_hat_matrix, S_hat_pinv)
            self.log.debug("after S_hat_pinv")
            

            # singular value decomposition of P
            # if P is symmetric and positive definite, the SVD is P = U d U.H instead of P = U d V.H
            U, d, V = np.linalg.svd(P)
            self.log.debug("after svd")

            # the transformation matrix A is then given by A = C U.H T
            # C is diagonal matrix with 1 on the first m rows and 0 in the remaining
            # instead of using C, only assing mxn to A
            C = np.array(np.zeros([n,n]), dtype=np.float32)
            self.log.debug("after C")
            for x in range(m):
                C[x,x]=1.
            A_square = np.dot(C, np.dot(U.T.conjugate(), T))
            A = A_square[0:m,:]
            self.log.debug("after A")

            # Compress the data
            if data.ndim == 5:
                out = np.zeros([m,extra_dim2,extra_dim1,nr_arms,nr_points],dtype=data.dtype)
                for extra2 in range(extra_dim2):
                    for extra1 in range(extra_dim1):
                        for arm in range(nr_arms):
                            for point in range(nr_points):
                                out[:,extra2,extra1,arm,point] = np.dot(A, data[:,extra2,extra1,arm,point])

            # SETTING PORT INFO
            self.setData('compressed data', np.squeeze(out))
            self.setData('A', A)
            self.setData('noise covariance', noise_cv_matrix)
    
            # end of compute
            if reset_compute_button:
                self.setAttr('compute', val=False)
    

        return 0

    def execType(self):
        '''Could be GPI_THREAD, GPI_PROCESS, GPI_APPLOOP'''
        return gpi.GPI_APPLOOP #gpi.GPI_PROCESS  #Mike-debug: does it need to be an apploop for display widget?
