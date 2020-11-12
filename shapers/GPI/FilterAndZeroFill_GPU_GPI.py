#   copyright
#   Author: Mike Schar
#   Date: 2018July11

import gpi
import numpy as np
import cupy as cp
class ExternalNode(gpi.NodeAPI):
    """
      Applies cylindrical inplane and 1D through plane Hanning Window filters in k-space, and zero-fills back to image domain.
      Input 
            Data in image domain
            User parameters (optional) to-do currently not used
      Output
            Filtered and zero-filled data in image domain
      Widgets

            
    """

    def initUI(self):
        # Widgets

        self.addWidget('PushButton', 'compute', toggle=True)
        self.addWidget('ExclusivePushButtons', 'Fourier interpolation factor', buttons=['none', '2', '4', '8'], val=1)  #, '8', '16'], val=1)
        self.addWidget('PushButton', 'Hann 2D Filter', toggle=True, button_title='ON', val=1)
        self.addWidget('Slider', 'Hann 2D Width (%)', val=90, min=0, max=100)
        self.addWidget('Slider', 'Hann 2D Taper (%)', val=30, min=0, max=100)
        self.addWidget('PushButton', 'Flip Image', toggle=True, button_title='ON', val=1)
        self.addWidget('PushButton', 'Hann 1D Filter', toggle=True, button_title='ON', val=1)
        self.addWidget('Slider', 'Hann 1D Width (%)', val=100, min=0, max=100)
        self.addWidget('Slider', 'Hann 1D Taper (%)', val=30, min=0, max=100)
        self.addWidget('Slider', 'Hann 1D StopVal (%)', val=5, min=0, max=100)

        # IO Ports
        # Kernel
        self.addInPort('data', 'NPYarray', dtype=[np.complex128, np.complex64, np.float32])
        self.addOutPort('Zero filled data', 'NPYarray')
        self.addOutPort('Hann 2D window', 'NPYarray', dtype=np.float32) # type from Kaiser2D_utils.py
        self.addOutPort('Hann 1D window', 'NPYarray', dtype=np.float32) 

    def validate(self):
        data = self.getData('data')
    
        doHanning2 = self.getVal('Hann 2D Filter')
        doHanning1 = self.getVal('Hann 1D Filter')

        # check dimensions
        if data.ndim != 3 and doHanning1:
            self.log.warn(" Error: # of dims of data must 3.")
            return 1
        
        if doHanning2:
            self.setAttr('Hann 2D Filter', button_title='ON')
            self.setAttr('Hann 2D Width (%)', visible=True)
            self.setAttr('Hann 2D Taper (%)', visible=True)
        else:
            self.setAttr('Hann 2D Filter', button_title='OFF')
            self.setAttr('Hann 2D Width (%)', visible=False)
            self.setAttr('Hann 2D Taper (%)', visible=False)
        if doHanning1:
            self.setAttr('Hann 1D Filter', button_title='ON')
            self.setAttr('Hann 1D Width (%)', visible=True)
            self.setAttr('Hann 1D Taper (%)', visible=True)
            self.setAttr('Hann 1D StopVal (%)', visible=True)
        else:
            self.setAttr('Hann 1D Filter', button_title='OFF')
            self.setAttr('Hann 1D Width (%)', visible=False)
            self.setAttr('Hann 1D Taper (%)', visible=False)
            self.setAttr('Hann 1D StopVal (%)', visible=False)

        return 0

    def compute(self):
        self.log.node("VesselSegmentation compute")

        # GETTING WIDGET INFO
        compute = self.getVal('compute')
        doHanning2 = self.getVal('Hann 2D Filter')
        Hann2_taper = self.getVal('Hann 2D Taper (%)')
        Hann2_width = self.getVal('Hann 2D Width (%)')
        Flip_image = self.getVal('Flip Image')
        doHanning1 = self.getVal('Hann 1D Filter')
        Hann1_taper = self.getVal('Hann 1D Taper (%)')
        Hann1_width = self.getVal('Hann 1D Width (%)')
        Hann1_stopVal = self.getVal('Hann 1D StopVal (%)')/100.0
        ZeroFill_factor_value = self.getVal('Fourier interpolation factor')
        
        if ZeroFill_factor_value == 0:
            ZeroFill_factor = 1
        elif ZeroFill_factor_value == 1:
            ZeroFill_factor = 2
        elif ZeroFill_factor_value == 2:
            ZeroFill_factor = 4
        elif ZeroFill_factor_value == 3:
            ZeroFill_factor = 8
        elif ZeroFill_factor_value == 4:
            ZeroFill_factor = 16
  
        # GETTING PORT INFO (1/2)
        #data = np.abs(self.getData('data'))  # not sure why one would do absolute here?
        data = cp.asarray( self.getData('data') )

        if ( compute and (ZeroFill_factor_value > 0 or doHanning1 or doHanning2) ):
            import triggeredGASSP.gridding.Kaiser2D_utils as kaiser2D
            
            # doHanning1 means through plane, assuming that the data is only 3D in this case
            if doHanning1:
                # FFT to k-space
                kSpace = cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(data, axes=(-3,-2,-1)), axes=(-3,-2,-1)), axes=(-3,-2,-1))

                # add Hanning filters 2D in-plane and 1D thru z to smooth out kspace before ZF
                if doHanning2:
                    self.log.node('Applying 2D Hann filter (in plane)')
                    # gridded k-space here
                    Hann2_win = cp.asarray( kaiser2D.window2(kSpace.shape[-2:], Hann2_taper, Hann2_width) )
                    for i_slc in range(kSpace.shape[0]):
                        kSpace[i_slc,...] *= Hann2_win
                        self.log.node(" in slice "+str(i_slc))
                if doHanning1:
                    self.log.node('Applying 1D Hann filter (thru slices)')
                    # gridded k-space here
                    # Hann1_width should always be 100% and stopVal > 0
                    Hann1_win = cp.asarray( kaiser2D.window1(kSpace.shape[-3], Hann1_taper, Hann1_width, stopVal=Hann1_stopVal) )
                    self.log.node("Hann1 shape: "+str(Hann1_win.shape))
                    Hann1_winTile = Hann1_win[:,cp.newaxis]
                    Hann1_winTile = Hann1_winTile[:,:,cp.newaxis]
                    Hann1_winTile = cp.tile(Hann1_winTile,(1,kSpace.shape[-2],kSpace.shape[-1]))
                    self.log.node("Hann1 tiled shape: "+str(Hann1_winTile.shape))
                    kSpace *= Hann1_winTile

                # zero filling based on FFT nodes
                for i in range(data.ndim):
                    zpad_length = ( ZeroFill_factor * data.shape[-i-1] ) - data.shape[-i-1]
                    if zpad_length >= 0:
                        zpad_before = int(zpad_length / 2.0 + 0.5)
                        zpad_after = int(zpad_length / 2.0)
                        temp = cp.insert(temp, data.shape[-i-1] * cp.ones(zpad_after), 0.0, (-i-1))
                        temp = cp.insert(temp, cp.zeros(zpad_before), 0.0, (-i-1))

                # often the image needs to be flipped to correspond to radiological orientation
                if Flip_image == 0:
                    out = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(temp), axes=(-3,-2,-1)))
                else:
                    out = cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(temp), axes=(-3,-2,-1)))
            else:
                # data dimensions
                nx = data.shape[-1]
                ny = data.shape[-2]
                if data.ndim == 2:
                    extra_dim1 = 1
                    extra_dim2 = 1
                elif data.ndim == 3:
                    extra_dim1 = data.shape[-3]
                    extra_dim2 = 1
                elif data.ndim == 4:
                    extra_dim1 = data.shape[-3]
                    extra_dim2 = data.shape[-4]
                elif data.ndim > 4:
                    self.log.warn("Only up to 4 dimensions implemented yet.")
                data.shape = [extra_dim2, extra_dim1, ny, nx]
                
                # GPU memory limitation - loop over cardiac phases - assumed to be extra_dim1
                out = np.zeros([extra_dim2, extra_dim1, ZeroFill_factor*ny, ZeroFill_factor*nx], dtype=data.dtype)
                if doHanning2:
                    self.log.node('Pre-calculate 2D Hann filter (in plane)')
                    Hann2_win = cp.asarray( kaiser2D.window2(data.shape[-2:], Hann2_taper, Hann2_width) )
                for extra_idx1 in range(extra_dim1):
                    # inverse FFT data in-plane only
                    kSpace = cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(data[:,extra_idx1,:,:], axes=(-1,-2)), axes=(-1,-2)), axes=(-1,-2))

                    if doHanning2:
                        self.log.node('Applying 2D Hann filter (in plane)')
                        for idx_dim2 in range(extra_dim2):
                            kSpace[idx_dim2,...] *= Hann2_win

                    # zero filling based on FFT nodes
                    if ZeroFill_factor_value == 0:
                        zero_filled_kSpace = kSpace
                    else:
                        zero_filled_kSpace = cp.zeros([extra_dim2, ZeroFill_factor*ny, ZeroFill_factor*nx], dtype=kSpace.dtype)
                        zpad_length = ( ZeroFill_factor * data.shape[-1] ) - data.shape[-1]
                        zpad_before = int(zpad_length / 2.0 + 0.5)
                        zpad_after = int(zpad_length / 2.0)
                        zero_filled_kSpace[:,zpad_before:-zpad_after,zpad_before:-zpad_after] = kSpace
                    
                    # often the image needs to be flipped to correspond to radiological orientation
                    if Flip_image == 0:
                        out[:,extra_idx1,:,:] = cp.asnumpy( cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(zero_filled_kSpace, axes=(-2,-1)), axes=(-2,-1)), axes=(-2,-1)) )
                    else:
                        out[:,extra_idx1,:,:] = cp.asnumpy( cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(zero_filled_kSpace, axes=(-2,-1)), axes=(-2,-1)), axes=(-2,-1)) )

            self.setData('Zero filled data', out.squeeze())
            if doHanning2 and Hann2_win is not None:
                self.setData('Hann 2D window', cp.asnumpy(Hann2_win))
            if doHanning1 and Hann1_win is not None:
                self.setData('Hann 1D window', cp.asnumpy(Hann1_win) )

        return 0
  
    
    def execType(self):
        '''Could be GPI_THREAD, GPI_PROCESS, GPI_APPLOOP'''
        return gpi.GPI_PROCESS

