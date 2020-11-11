# Author: Nick Zwart
# Date: 2016feb18

import numpy as np
import gpi

# This is a template node, with stubs for initUI() (input/output ports,
# widgets), validate(), and compute().
# Documentation for the node API can be found online:
# http://docs.gpilab.com/NodeAPI

class ExternalNode(gpi.NodeAPI):
    """FOVShift uses the coordinates and input shift arguments to add a linear
    phase to the data.  If dictionary header is provided, shifts are calculated 
    and gui input values ignored. For 2D and 3D data.

    INPUT:
        crds - 2-vec or 3-vec array
        data - raw k-space data corresponding to crds
        params_in - dictionary data header (optional) 
    OUTPUT:
        adjusted data - original k-space data with linear phase
    WIDGETS:
        dx,dy,dz - FOV shift in pixels
    """

    # initialize the UI - add widgets and input/output ports
    def initUI(self):
        # Widgets
        self.addWidget('SpinBox','Eff MTX XY', min=5, val=240)
        self.addWidget('SpinBox','Eff MTX Z',  min=5, val=240)
        self.addWidget('DoubleSpinBox', 'dx (pixels)', val=0)
        self.addWidget('DoubleSpinBox', 'dy (pixels)', val=0)
        self.addWidget('DoubleSpinBox', 'dz (pixels)', val=0)

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=[np.complex64,np.complex128])
#        self.addInPort('crds', 'NPYarray', dtype=[np.float32,np.float64], vec=2)
        self.addInPort('crds', 'NPYarray', dtype=[np.float32,np.float64])
        self.addInPort('params_in', 'DICT', obligation = gpi.OPTIONAL)    
        
        self.addOutPort('adjusted data', 'NPYarray', dtype=[np.complex64,np.complex128])


    def validate(self):
        crds = self.getData('crds')
        inparam = self.getData('params_in')

        if crds.shape[-1] == 3:
            self.log.node("*** 3D data")                                                       
        else:
            self.setAttr('dz (pixels)', visible=False)
            self.setAttr('Eff MTX Z', visible=False)
            self.log.node("*** 2D data")

        # GB: if dict given as input compute dx,dy,dz (as in Grid_GPI.py) 
        if (inparam is not None):                                               

          # header check
          if 'headerType' in inparam:

            # check if the header is for spiral
            if inparam['headerType'] != 'BNIspiral':
                self.log.warn("wrong header type")
                return 1

          else:
            # if there is no header type, then its also the wrong type
            self.log.warn("wrong header type")
            return 1

          # Auto Matrix calculation: extra 25% assumes "true resolution"          
          mtx_xy = 1.25*float(inparam['spFOVXY'][0])/float(inparam['spRESXY'][0])
          self.setAttr('Eff MTX XY', val = mtx_xy)                   
          self.log.node("*** Eff MTX XY "+str(mtx_xy))        

          if crds.shape[-1] == 3:                                               
            mtx_z  = float(inparam['spFOVZ'][0]) /float(inparam['spRESZ'][0])   
            if int(float(inparam['spSTYPE'][0])) in [2,3]: #SDST, FLORET        
              mtx_z *= 1.25
            self.setAttr('Eff MTX Z', val = mtx_z)                   
            self.log.node("*** Eff MTX Z "+str(mtx_z))                                                         

          # Auto offset calculation.  Values reported in mm, change to # pixels
          m_off = 0.001*float(inparam['m_offc'][0])
          p_off = 0.001*float(inparam['p_offc'][0])
          xoff = m_off*float(mtx_xy)/float(inparam['spFOVXY'][0])
          yoff = p_off*float(mtx_xy)/float(inparam['spFOVXY'][0])
          self.setAttr('dx (pixels)', val=xoff)
          self.setAttr('dy (pixels)', val=yoff)
          self.log.node("*** Computed off-centers dx="+str(xoff)+" dy="+str(yoff))  
 
          if crds.shape[-1] == 3:                                               
            s_off = 0.001*float(inparam['s_offc'][0])
            zoff = s_off*float(mtx_z) /float(inparam['spFOVZ'][0])
            # shift half pixel when the number of slices is even with
            # distributed spirals. ZQL
            if (int(float(inparam['spSTYPE'][0])) in [1,2]) and \
              (int(mtx_z)%2 == 0):
              zoff = zoff - 0.5
            self.setAttr('dz (pixels)', val=zoff)
            self.log.node("*** Computed off-center dz="+str(zoff))   
 


    # process the input data, send it to the output port
    # return 1 if the computation failed
    # return 0 if the computation was successful 
    def compute(self):
        data = self.getData('data')
        crds = self.getData('crds')

        dx = self.getVal('dx (pixels)')
        dy = self.getVal('dy (pixels)')
        if crds.shape[-1] == 3:
            dz = self.getVal('dz (pixels)')
            phase = np.exp(-1j * 2.0 * np.pi * (crds[...,0]*dx + crds[...,1]*dy + crds[...,2]*dz))
            self.log.node("*** Computed 3D phase shift")   
        else:
            phase = np.exp(-1j * 2.0 * np.pi * (crds[...,0]*dx + crds[...,1]*dy))
            self.log.node("*** Computed 2D phase shift")   

        self.setData('adjusted data', data * phase)

        return 0
