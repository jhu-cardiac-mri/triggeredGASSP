# Author and copyright: Mike Schar
# Date: 2015Oct14
# updated by Gabriele Bonanno for sharpness measurements
# updated by Dan Zhu for flow data

import gpi
import numpy as np
from gpi import QtCore, QtGui

class ExternalNode(gpi.NodeAPI):
    """
      Autofocus of coronary spiral images
      Only the area of the coronary is corrected.
      Requires water only images (excited with spectra-spatial RF pulse).
      Input 
            Water images in 2D, multiple cardiac phases are allowed.
            Coordinates of spiral acquisition
            Coordinates of coronary artery in image.
            User parameters (optional)
            A second, identical set of 2D images, for phantom snr measurement when neighboring cardiac phases have correlated noise
      Output
            Image with deblurred coronary artery.
      Widgets
            data type: Select 2D cine in-vivo data or phantom data with multiple holes to analyze.
            phantom: In case data type is phantom, this widget is enabled to select which phantom was imaged:
                Select either original JHU phantom, CHUV 3mm phantom, or CHUV 4 mm phantom.
                Three Points need to be chosen:
                - for JHU phantom: first the vessel in the middle of one edge that is shifted inwards.
                                    2nd, the bigger corner vessel on that same edge.
                                    3rd, the small corner vessel on the other edge from where the 2nd point was.
                - for CHUV 3mm phantom: first click on vessel next to cut corner.
                                    2nd click on vessel in adjacent corner that is 10 vessels away.
                                    3rd, click on vessel in corner adjacent to first vessel that is 11 vessels away.
                - for CHUV 4mm phantom: first click on vessen next to cut corner.
                                    2nd click on vessel in adjacent corner that is 10 vessels away.
                                    3rd, click on vessel in corner adjacent to first vessel that is 15 vessels away.
                - for JHU test phantom: first click on top left vessel.
                                    2nd on vessel below.
                                    3rd on vessel top right.
                - for Swiss Cheese: first the vessel that is out of line and closer to another vessel than any other one.
                                    2nd click on the bigger corner vessel on that same edge.
                                    3rd, the small corner vessel on the other edge from where the 2nd point was.
            image: Shows image of the cardiac phase chosen in the cardiac phase slider. The location of autofocus is selected in this image with a left mouse button click.
            image ceiling: Slider to adjust the image brighness ceiling for the display box widget above.
            cardiac phase: Slider to select the cardiac phase to be analyzed. The chosen phase is displayed in the image display box widget above.
            compute: Push button to enable computation.
            adc (ms); float: Duration of acquisition window; tau
                set from user parameters if params_in is connected
            focus radius (pixels); int: radious from the center of the vessel used in algorithm
            frequency range (Hz); int: plus to minus this frequency will be tested.
            number steps; int: number of frequency steps
            cardiac phases: the number of cardiac phases to be analyzed centered around the chosen phase in the slider 'cardiac phase'.
            
    """

    def initUI(self):
        # Widgets
        self.addWidget('ExclusivePushButtons', 'data type', buttons=['2D cine', 'phantom'], val=0)
        self.addWidget('ExclusivePushButtons', 'phantom',
                       buttons=['JHU_1', 'CHUV 3mm', 'CHUV 4mm', 'JHU_test', 'Swiss Cheese'],
                       val=4, visible=False)
        self.addWidget('ExclusivePushButtons', 'perform..', buttons=['deblur and analyze', 'analyze'], val=0)
        self.addWidget('DisplayBox', 'image', interp=True, ann_box='Pointer')
        self.addWidget('DisplayBox', 'results', interp=True)
        self.addWidget('Slider', 'FWHM slider 50 55 60 65 70 75 80', min=1, max=7, val=1)
        self.addWidget('TextBox', 'Areas:', visible=False)
        self.addWidget('Slider', 'image ceiling',val=60)
        self.addWidget('Slider', 'cardiac phase')
        self.addWidget('PushButton', 'compute', toggle=True)
        self.addWidget('DoubleSpinBox', 'adc (ms)', val=10, min=0.01, max=100)
        self.addWidget('SpinBox', 'focus radius (mm)', val=5)
        self.addWidget('SpinBox', 'frequency range (Hz)', val=100)
        self.addWidget('SpinBox', 'frequency steps', min=1, val=11, singlestep=2)
        self.addWidget('ExclusivePushButtons', 'cardiac phases type', buttons=['all', 'range'], val=0)
        self.addWidget('SpinBox', 'cardiac phases', val=3, min=1)
        self.addWidget('PushButton', 'FWHM area', toggle=True, val=False)
        self.addWidget('SpinBox', 'FOV (mm)', val=220, min=1)
        self.addWidget('ExclusivePushButtons', 'Fourier interpolation factor', buttons=['none', '2', '4', '8', '16'], val=3)

        # IO Ports
        # Kernel
        self.addInPort('data', 'NPYarray', dtype=[np.complex128, np.complex64, np.float32])
        self.addInPort('coords', 'NPYarray', vec=2, obligation = gpi.OPTIONAL)
        self.addInPort('params_in', 'DICT', obligation = gpi.OPTIONAL)
        self.addInPort('filter', 'NPYarray', dtype=[np.float32, np.float64], obligation = gpi.OPTIONAL)
        self.addInPort('data reacquired', 'NPYarray', dtype=[np.complex128, np.complex64, np.float32], obligation = gpi.OPTIONAL)
        
        data = self.getData('data')
        self.addOutPort('deblurred data', 'NPYarray', ) #dtype=[np.complex128, np.complex64])
        self.addOutPort('deblurred data flow', 'NPYarray', ) #dtype=[np.complex128, np.complex64])    
        self.addOutPort('Cardiac Phase, vessel, FWHM area, sharpness, freq, snr', 'NPYarray', )
        self.addOutPort('blurred array', 'NPYarray', ) #dtype=[np.complex128, np.complex64])
        self.addOutPort('blurred array flow', 'NPYarray', ) #dtype=[np.complex128, np.complex64])   
        self.addOutPort('imaginary integral for deblurring', 'NPYarray', )
        self.addOutPort('debug', 'NPYarray')
        self.addOutPort('FWHM_array', 'NPYarray')
        self.addOutPort('debug3', 'NPYarray')
        self.addOutPort('phantom_area', 'NPYarray')
        self.addOutPort('phantom_sharpness', 'NPYarray')
        self.addOutPort('phantom_freq', 'NPYarray')
        self.addOutPort('phantom_snr', 'NPYarray')
        self.addOutPort('kspace', 'NPYarray')
        self.addOutPort('Debug Data', 'NPYarray')
    
        self.points_selected = 0
        self.points = []

    def validate(self):
        data = self.getData('data')
        coords = self.getData('coords')
        params = self.getData('params_in')
        data_reacquired = self.getData('data reacquired')
        
        if (self.getVal('perform..') == 0):
            if coords is None:
                self.log.warn("Deblurring requires trajectory coordinates.")
                return 1
            if (data.dtype == np.float32):
                self.log.warn("Deblurring requires complex data.")
                return 1
        
        if params is not None:
            if 'headerType' in params:
                if params['headerType'] == 'BNIspiral': # BNI spiral raw data
                    if 'spDWELL' in params:
                        dwell = float(params['spDWELL'][0])
                        adc = dwell * coords.shape[-2]
                        self.setAttr('adc (ms)', val=adc)
                    if 'spFOVXY' in params:
                        FOV_float = float(params['spFOVXY'][0])
                        FOV = int(1000.*FOV_float)
                        self.setAttr('FOV (mm)', val=FOV)
                elif params['headerType'] == '.xml': # xml/rec data
                    if ( ('Pixel Spacing' in params) and ('Resolution X' in params) ):
                        orig_voxel_size_string = params['Pixel Spacing'][0]
                        orig_voxel_size = float(orig_voxel_size_string.split()[0])
                        orig_matrix = float(params['Resolution X'][0])
                        FOV_float = orig_voxel_size * orig_matrix
                        FOV = int(1.*FOV_float)
                        self.setAttr('FOV (mm)', val=FOV)
                elif params['headerType'] == '.par': # par/rec data
                    #self.log.debug('Getting FOV information from .par file')
                    #self.log.debug( ('Pixel Spacing_1' in params) )
                    #self.log.debug( ('Resolution X' in params) )
                    if ( ('Pixel Spacing_1' in params) and ('Resolution X' in params) ):
                        #self.log.debug('in if statement')
                        orig_voxel_size = float( params['Pixel Spacing_1'][0] )
                        #self.log.debug(orig_voxel_size)
                        orig_matrix = float(params['Resolution X'][0])
                        #self.log.debug(orig_matrix)
                        FOV_float = orig_voxel_size * orig_matrix
                        #self.log.debug(FOV_float)
                        FOV = int(1.*FOV_float)
                        #self.log.debug(FOV)
                        self.setAttr('FOV (mm)', val=FOV)
                        #self.log.debug(FOV,orig_matrix,orig_voxel_size)
   
        # Phantom selection
        if self.getVal('data type') == 1:
            self.setAttr('phantom', visible=True)
        else:
            self.setAttr('phantom', visible=False)
  
        # check dimensions
        
        # Modification Starts Here: Dan Zhu
        # New Codes
         # check dimensions
        if data.ndim < 2 or data.ndim > 4:
            print(" Error: # of dims of data must be 2 or 4.")
            return 1       
         
        if data.ndim == 4:
            if data.shape[0]!=2: # Now we only accept flow encodes=2
                print(" Error: # of dims of data must be 2.")
                return 1
            
        # Modification Ends Here: Dan Zhu
        
        # keep deblurring frequences symmetric around zero, therefore, steps should be odd
        nr_freq = self.getVal('frequency steps')
        if nr_freq % 2 == 0:
            self.setAttr('frequency steps', val=nr_freq + 1)
        
        
        # cardiac phases
        if data.ndim == 3:
            self.setAttr('cardiac phase', min=1,max=data.shape[0])
            self.setAttr('cardiac phases', min=1,max=data.shape[0])
        
            # limit cardiac phases according to selected cardiac phase.
            cardiac_phases = self.getVal('cardiac phases')
            cardiac_phase = self.getVal('cardiac phase')
            if (cardiac_phase + int(np.floor(0.5*cardiac_phases)) > data.shape[0]):
                self.log.node("Reduce number of chosen cardiac phases because the end of the cardiac cycle has been reached.")
                self.setAttr('cardiac phases', val=1 + 2*(data.shape[0]-cardiac_phase) )
            if (cardiac_phase < int(np.ceil(0.5*cardiac_phases))):
                self.log.node("Reduce number of chosen cardiac phases because the beginning of the cardiac cycle has been reached.")
                self.setAttr('cardiac phases', val=2*(cardiac_phase) )
        
        # Modification Starts Here: Dan Zhu
        # Codes Added:
        elif data.ndim == 4:          
            self.setAttr('cardiac phase', min=1,max=data.shape[1])
            self.setAttr('cardiac phases', min=1,max=data.shape[1])
        
            # limit cardiac phases according to selected cardiac phase.
            cardiac_phases = self.getVal('cardiac phases')
            cardiac_phase = self.getVal('cardiac phase')
            if (cardiac_phase + int(np.floor(0.5*cardiac_phases)) > data.shape[1]):
                self.log.node("Reduce number of chosen cardiac phases because the end of the cardiac cycle has been reached.")
                self.setAttr('cardiac phases', val=1 + 2*(data.shape[1]-cardiac_phase) )
            if (cardiac_phase < int(np.ceil(0.5*cardiac_phases))):
                self.log.node("Reduce number of chosen cardiac phases because the beginning of the cardiac cycle has been reached.")
                self.setAttr('cardiac phases', val=2*(cardiac_phase) )
        # Modification Ends Here: Dan Zhu
        
        else:
            self.setAttr('cardiac phase', min=1,max=1)
            self.setAttr('cardiac phases', min=1,max=1,val=1)

        # FWHM area determination
        cardiac_type = self.getVal('cardiac phases type')
        if cardiac_type == 0:
            self.setAttr('FWHM area', val=False, visible=False)
            self.setAttr('cardiac phases', visible=False)
        else:
            self.setAttr('FWHM area', visible=True)
            self.setAttr('cardiac phases', visible=True)
        FWHM_area = self.getVal('FWHM area')
        if FWHM_area:
            self.setAttr('FOV (mm)', visible=True)
            self.setAttr('Fourier interpolation factor', visible=True)
        else:
            self.setAttr('FOV (mm)', visible=False)
            self.setAttr('Fourier interpolation factor', visible=False)

        debug = self.getData('debug')
        if debug is not None:
            self.setAttr('results', visible=True)
            self.setAttr('FWHM slider 50 55 60 65 70 75 80', visible=True)
        else:
            self.setAttr('results', visible=False)
            self.setAttr('FWHM slider 50 55 60 65 70 75 80', visible=False)
            self.setAttr('FWHM slider 50 55 60 65 70 75 80', value=1)
        
        # select points
        if 'image' in self.widgetEvents():
            if self.points_selected == 0:
                self.points += [np.array(self.getAttr('image', 'points'), dtype='int32')]
                self.points_selected = 1
            else:
                if self.getVal('data type') == 0:
                    # replace current point
                    self.points = [np.array(self.getAttr('image', 'points'), dtype='int32')]
                else:
                    # add up to 3 points
                    if self.points_selected < 3:
                        self.points += [np.array(self.getAttr('image', 'points'), dtype='int32')]
                        self.points_selected += 1
                    else:
                        self.points[0] = self.points[1]
                        self.points[1] = self.points[2]
                        self.points[2] = np.array(self.getAttr('image', 'points'), dtype='int32')

        # 2nd identical acquisition for SNR measurements (for instance in case neighboring cardiac phases are not independent)
        if data_reacquired is not None:
            if data.shape != data_reacquired.shape:
                self.log.warn("Reacquired data needs to have the same size as data.")
                return 1
            if data.ndim == 3:
                phases = 2 * data.shape[0]
                        # Modification Starts Here: Dan Zhu
            # Codes Added:
            elif data.ndim == 4:
                phases = 2 * data.shape[1]
            # Modification Ends Here: Dan Zhu
            else:
                phases = 2
            self.setAttr('cardiac phases', min=1,max=phases)
            self.setAttr('cardiac phases', val=phases )

        return 0

    def compute(self):
        self.log.node("AutoFocus compute")

# GETTING WIDGET INFO
        data_type = self.getVal('data type')
        image_ceiling = self.getVal('image ceiling')
        cardiac_phase = self.getVal('cardiac phase')
        compute = self.getVal('compute')
        adc = self.getVal('adc (ms)')
        radius_mm = self.getVal('focus radius (mm)')
        delta_f0 = self.getVal('frequency range (Hz)')
        nr_freq = self.getVal('frequency steps')
        if self.getVal('perform..') == 0:
            doDeblur = True
        else:
            doDeblur = False
            self.setAttr('cardiac phases type', val = 1)
            nr_freq = 1
        cardiac_type = self.getVal('cardiac phases type')
        cardiac_phases = self.getVal('cardiac phases')
        FWHM_area = self.getVal('FWHM area')
        FOV = self.getVal('FOV (mm)')
        ZeroFill_factor_value = self.getVal('Fourier interpolation factor')
        if FWHM_area:
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
 
        # move below to distinguish between in-vivo data and phantom nr_FWHM_steps = 7
        if data_type == 0:
            nr_FWHM_steps = 7
        else:
            nr_FWHM_steps = 1
        FWHM_step_size = 5.

        # display current cardiac phase
        # GETTING PORT INFO (1/2)
        data = self.getData('data')
        
        if data.ndim == 3:
            phases = data.shape[0]
        # modification starts here: Dan Zhu
        elif data.ndim ==4:
            phases = data.shape[1]
            flow_encodes = data.shape[0]
            data_flow=np.copy(data)
            data=np.sqrt(data[0,...]*data[1,...]+0*1j)
        # modification ends here: Dan Zhu
        else:
            phases = 1
            data = data.reshape([1,data.shape[-2],data.shape[-1]])
        image = np.abs(data[cardiac_phase-1,:,:])

        # modification starts here: Dan Zhu
        import triggeredGASSP.gridding.Kaiser2D_utils as kaiser2D
        mask=np.ones(data.shape)

        # modification ends here: Dan Zhu
        
        filter = self.getData('filter')
        if filter is not None:
            has_filter = True
        else:
            has_filter = False

        
        # reacquired data, for phantom snr measurement when neighboring cardiac phases have correlated noise
        has_reacquired_data = 0
        data_reacquired = self.getData('data reacquired')
        if data_reacquired is not None:
            has_reacquired_data = 1
            phases *= 2
            cardiac_phase = phases//2
            backup = data.copy()
            data = np.zeros([phases, data.shape[-2], data.shape[-1]], dtype=data.dtype)
            for pha in range(phases//2):
                data[2*pha,...] = backup[pha,...]
                data[2*pha+1,...] = data_reacquired[pha,...]
        
        data_min = image.min()
        data_max = image.max()
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
        
        # Add green circle for point location with size of chosen area
        # For phantom analysis add red circle for 2nd and 3rd points.
        if self.points_selected > 2:
            c_coords = self.points[2]
            image1[c_coords[1]-3,c_coords[0]-1:c_coords[0]+2,:] = [255,0,0,1]
            image1[c_coords[1]+3,c_coords[0]-1:c_coords[0]+2,:] = [255,0,0,1]
            image1[c_coords[1]-1:c_coords[1]+2,c_coords[0]-3,:] = [255,0,0,1]
            image1[c_coords[1]-1:c_coords[1]+2,c_coords[0]+3,:] = [255,0,0,1]
            image1[c_coords[1]-2,c_coords[0]-2,:] = [255,0,0,1]
            image1[c_coords[1]-2,c_coords[0]+2,:] = [255,0,0,1]
            image1[c_coords[1]+2,c_coords[0]+2,:] = [255,0,0,1]
            image1[c_coords[1]+2,c_coords[0]-2,:] = [255,0,0,1]
        if self.points_selected > 1:
            c_coords = self.points[1]
            image1[c_coords[1]-3,c_coords[0]-1:c_coords[0]+2,:] = [255,0,0,1]
            image1[c_coords[1]+3,c_coords[0]-1:c_coords[0]+2,:] = [255,0,0,1]
            image1[c_coords[1]-1:c_coords[1]+2,c_coords[0]-3,:] = [255,0,0,1]
            image1[c_coords[1]-1:c_coords[1]+2,c_coords[0]+3,:] = [255,0,0,1]
            image1[c_coords[1]-2,c_coords[0]-2,:] = [255,0,0,1]
            image1[c_coords[1]-2,c_coords[0]+2,:] = [255,0,0,1]
            image1[c_coords[1]+2,c_coords[0]+2,:] = [255,0,0,1]
            image1[c_coords[1]+2,c_coords[0]-2,:] = [255,0,0,1]
        if self.points_selected > 0:
            c_coords = self.points[0]
            image1[c_coords[1]-3,c_coords[0]-1:c_coords[0]+2,:] = [0,255,0,1]
            image1[c_coords[1]+3,c_coords[0]-1:c_coords[0]+2,:] = [0,255,0,1]
            image1[c_coords[1]-1:c_coords[1]+2,c_coords[0]-3,:] = [0,255,0,1]
            image1[c_coords[1]-1:c_coords[1]+2,c_coords[0]+3,:] = [0,255,0,1]
            image1[c_coords[1]-2,c_coords[0]-2,:] = [0,255,0,1]
            image1[c_coords[1]-2,c_coords[0]+2,:] = [0,255,0,1]
            image1[c_coords[1]+2,c_coords[0]+2,:] = [0,255,0,1]
            image1[c_coords[1]+2,c_coords[0]-2,:] = [0,255,0,1]

        format_ = QtGui.QImage.Format_RGB32
        
        image2 = QtGui.QImage(image1.data, w, h, format_)
        image2.ndarry = image1
        self.setAttr('image', val=image2)


        #self.log.debug("AutoFocus compute - before if compute")
        if (compute and ( (self.points_selected > 0 and data_type == 0) or (self.points_selected == 3 and data_type == 1) ) ):
            import triggeredGASSP.spiral.autofocus as af

            # GETTING PORT INFO (2/2)
            coords = self.getData('coords')

            params = self.getData('params_in')
            
            matrix_size = data.shape[-1]
            voxel_size = FOV / matrix_size
            nr_radii = (int)(1. * radius_mm / voxel_size)
            print("matrix " + str(matrix_size) + ", voxel_size " + str(voxel_size) + ", nr_radii " + str(nr_radii) + ". \n")
            
            
            # multiple location points for phantom analysis
            if data_type == 0:
                nr_of_points = 1
                nr_iterations = 2 # to track the vessel in real data
                points_phantom = np.zeros([nr_of_points,2], dtype = np.int16)
                points_phantom[0,:] = self.points[0]
            else:
                # use code from phantomFWHM node to select which phantom was chosen
                phantom = self.getVal('phantom')
                if phantom == 0:
                    nr_of_points = 42
                    #mask_radius = 6 # mm
                elif phantom == 1:
                    nr_of_points = 110
                    nr_rows = 10
                    nr_columns = 11
                    #mask_radius = 4 #mm
                elif phantom == 2:
                    nr_of_points = 150
                    nr_rows = 10
                    nr_columns = 15
                    #mask_radius = 4 #mm
                elif phantom == 3:
                    nr_of_points = 6
                    nr_rows = 2
                    nr_columns = 3
                    #mask_radius = 5 #mm
                elif phantom == 4:
                    nr_of_points = 142
                    nr_rows = 13
                    nr_columns = 11
                    #mask_radius = 5 #mm
                
                nr_iterations = 2 # do not try to track data in phantom
                if phantom == 0:
        
                  # three points to click are Point24, PointTwo, PointThree
                  Point24 = self.points[0]
                  PointTwo = self.points[1]
                  PointThree = self.points[2]

                  # create two vectors between the 3 points and determine the angle between them to determine if phantom was flipped
                  vec1 = Point24 - PointTwo
                  vec2 = PointThree - PointTwo

                  if vec1[1] == 0:
                    alpha1 = np.pi / 2.
                  else:
                    alpha1 = np.arctan(1.*vec1[0]/vec1[1])
                  if vec2[1] == 0:
                    alpha2 = np.pi / 2.
                  else:
                    alpha2 = np.arctan(1.*vec2[0]/vec2[1])

                  angle = (alpha2-alpha1) / np.pi * 180

                  # find coordinate for all points
                  points_float = np.zeros((nr_of_points,2))

                  if (angle > 70.) and (angle < 80):
                    Point1 = PointThree + np.array([-vec2[1],vec2[0]])
                    # to move to the next row move down by md
                    md = -1. / 6. * np.array([-vec2[1],vec2[0]])
                  else:
                    Point1 = PointThree + np.array([vec2[1],-vec2[0]])
                    # to move to the next row move down by md
                    md = -1. / 6. * np.array([vec2[1],-vec2[0]])

                  # rows 1-3 and 4-7 move to the right by step mr
                  mr = -0.2 * vec2
                  # row 4 move to the right by step mr4
                  mr4 = Point24 - ( Point1 + 3 * md)  # total distance in pixels
                  mr4_steps = 1. / 109 * np.array( [0, 17.5, 38., 63., 89., 109.])  #normalized steps size base on actual distance in mm

                  for col in range(6):
                    for row in range (7):
                        point_nr = (row*6) + col
                        if row == 3: #not equally distributed
                            points_phantom[point_nr] = Point1 + mr4_steps[col] * mr4 + row * md
                        else:
                            points_phantom[point_nr] = Point1 + col * mr + row * md

                elif ((phantom == 1) or (phantom==2)):
                  # three points to click are PointOne, PointHundred, and PointEleven
                  PointOne = self.points[0]
                  PointHundred = self.points[1]
                  PointEleven = self.points[2]

                  # create two vectors between the 3 points and determine the angle between them to determine if phantom was flipped
                  vec1 = PointHundred - PointOne
                  vec2 = PointEleven - PointOne
                  row_step = 1. / (nr_rows-1) * np.array([vec1[0],vec1[1]])
                  column_step = 1. / (nr_columns-1) * np.array([vec2[0], vec2[1]])
                  
                  # find coordinate for all points
                  points_phantom = np.zeros((nr_of_points,2))
                      
                  for row in range (nr_rows):
                    for col in range(nr_columns):
                        point_nr = (row * nr_columns) + col
                        point_float = PointOne + col * column_step + row * row_step
                        points_phantom[point_nr, 0] = point_float[0]
                        points_phantom[point_nr, 1] = point_float[1]

                elif (phantom == 3):
                  # three points to click are PointOne, PointHundred, and PointEleven
                  PointOne = self.points[0]
                  PointHundred = self.points[1]
                  PointEleven = self.points[2]

                  # create two vectors between the 3 points and determine the angle between them to determine if phantom was flipped
                  vec1 = PointHundred - PointOne
                  vec2 = PointEleven - PointOne
                  row_step = 1. / (nr_rows-1) * np.array([vec1[0],vec1[1]])
                  column_step = 1. / (nr_columns-1) * np.array([vec2[0], vec2[1]])
                  
                  # find coordinate for all points
                  points_phantom = np.zeros((nr_of_points,2))
                      
                  for row in range (nr_rows):
                    for col in range(nr_columns):
                        point_nr = (row * nr_columns) + col
                        point_float = PointOne + col * column_step + row * row_step
                        points_float[point_nr, 0] = point_float[0]
                        points_float[point_nr, 1] = point_float[1]

                elif (phantom == 4):
                    # mask_radius = 5 #mm
                    # three points to click are Point24, PointTwo, PointThree
                    Point24 = self.points[0]
                    PointTwo = self.points[1]
                    PointThree = self.points[2]
                    
                    # create two vectors between the 3 points and determine the angle between them to determine if phantom was flipped
                    vec1 = Point24 - PointTwo
                    vec2 = PointThree - PointTwo
                    
                    if vec1[1] == 0:
                        alpha1 = np.pi / 2.
                    else:
                        alpha1 = np.arctan(1.*vec1[0]/vec1[1])
                    if vec2[1] == 0:
                        alpha2 = np.pi / 2.
                    else:
                        alpha2 = np.arctan(1.*vec2[0]/vec2[1])
                    
                    angle = (alpha2-alpha1) / np.pi * 180
                    
                    if ( ((angle > 70.) and (angle < 80)) or ((angle > -110.) and (angle < -100)) ):
                        PointOne = PointTwo + np.array([-vec2[1],vec2[0]])
                        row_dist =  -np.array([-vec2[1],vec2[0]])
                    else:
                        PointOne = PointTwo + np.array([vec2[1],-vec2[0]])
                        row_dist =  -np.array([vec2[1],-vec2[0]])
                    
                    row_step = 1. / (nr_rows-1) * row_dist
                    column_step = 1. / (nr_columns-1) * np.array([vec2[0], vec2[1]])
                    
                    # find coordinate for all points
                    points_phantom = np.zeros([nr_of_points,2], dtype = np.int16)
                    point_nr = 0
                    
                    for row in range (nr_rows):
                        if row == 6:
                            point_float = PointOne + row * row_step
                            points_phantom[point_nr, 0] = np.round(point_float[0])
                            points_phantom[point_nr, 1] = np.round(point_float[1])
                            point_nr += 1
                            for col in range(9):
                                if (col < 1 or col > 6):
                                    special_column_step = 1.5 * column_step
                                elif (col == 1):
                                    special_column_step = 0.5 * column_step
                                else:
                                    special_column_step = column_step
                                point_float = point_float + special_column_step
                                points_phantom[point_nr, 0] = np.round(point_float[0])
                                points_phantom[point_nr, 1] = np.round(point_float[1])
                                point_nr += 1
                        else:
                            for col in range(nr_columns):
                                point_float = PointOne + col * column_step + row * row_step
                                points_phantom[point_nr, 0] = np.round(point_float[0])
                                points_phantom[point_nr, 1] = np.round(point_float[1])
                                point_nr += 1

            # Deblurring
            if doDeblur:
            
                # setting up parameters
                # generate time vs. k-space radius
                nr_samples = coords.shape[-2]


                # Determine off-resonance induced phase in k-space
                # generate time vs. k-space radius
                k_space_radius = np.zeros(nr_samples)
                dwell = adc / nr_samples
                time = np.float32(list(range(nr_samples))) * dwell
                for ind in range(nr_samples):
                    k_space_radius[ind] = np.sqrt( coords[0,ind,0]**2 + coords[0,ind,1]**2 )

                k_space_radius_array = np.zeros(matrix_size**2)
                for x in range(matrix_size):
                    for y in range(matrix_size):
                        index = y + (x * matrix_size)
                        k_space_radius_array[index] = np.sqrt( ( 1. / matrix_size * ( x - (matrix_size / 2.)) )**2 + ( 1. / matrix_size * ( y - (matrix_size / 2.)) )**2 )

                #self.log.debug("AutoFocus compute - before looping over frequences")
                # loop over all requested deblurring frequences
                f0_step = 2.* delta_f0 / (nr_freq - 1)
                rot_matrix = np.array( np.zeros([nr_freq,matrix_size,matrix_size]), dtype=np.complex64)
                for freq_ind in range(nr_freq):
                    f0 = -delta_f0 + ( freq_ind * f0_step )

                    # off-resonance delta_f0 [Hz] and delta_omega [rad]
                    delta_omega = 2. * np.pi * f0
                
                    # generate off-resonance induce phase vs. k-space radious
                    phase = np.zeros(nr_samples)
                    phase = time * 0.001 * delta_omega
                
                    phase_array = np.interp( k_space_radius_array, k_space_radius, phase)
                    phase_matrix = phase_array.reshape( (matrix_size, matrix_size) )
                    rot_matrix[freq_ind,:,:] = np.exp(1j * phase_matrix)
                
                if cardiac_type == 0:
                    deblurred_data = np.zeros([phases, matrix_size, matrix_size ], dtype=np.complex64 )
                    blurred_data_debug = np.zeros([1, nr_freq, matrix_size, matrix_size ], dtype=np.complex64 )
                    # modification starts here: Dan Zhu
                    if flow_encodes > 1:
                        deblurred_data_flow = np.zeros([flow_encodes,phases, matrix_size, matrix_size ], dtype=np.complex64 )
                        blurred_data_debug_flow=np.zeros([flow_encodes,1, nr_freq, matrix_size, matrix_size ], dtype=np.complex64 )
                    # modification ends here: Dan Zhu
                    
                    cardiac_phases = 1
                else:
                    deblurred_data = np.zeros([cardiac_phases, matrix_size, matrix_size ], dtype=np.complex64 )
                    blurred_data_debug = np.zeros([cardiac_phases, nr_freq, matrix_size, matrix_size ], dtype=np.complex64 )
                    # modification starts here: Dan Zhu                  
                    if flow_encodes > 1:
                        deblurred_data_flow = np.zeros([flow_encodes,phases, matrix_size, matrix_size ], dtype=np.complex64 )
                        blurred_data_debug_flow=np.zeros([flow_encodes,cardiac_phases, nr_freq, matrix_size, matrix_size ], dtype=np.complex64 )
                    # modification ends here: Dan Zhu
                
                # autofocus method by Man L-C. MRM 37 p 906, 1997
                intergral_sqrt_imaginary_area = np.zeros([cardiac_phases,nr_of_points,nr_freq])

                deblur_freq = np.zeros([cardiac_phases, nr_of_points], dtype=np.int16)
                
                # number of pixels above FWHM in mask, and associated cardiac phase in cine and offset from chosen cardiac phase (cardiac_phase)
                pixels_per_mask = np.zeros([cardiac_phases, nr_of_points, 10])

                # loop over chosen cardiac phases
                for cardiac_phase_ind in range(cardiac_phases):
                    additional_step = 1 - int(np.ceil(0.5*cardiac_phases)) + cardiac_phase_ind
                    current_cardiac_phase = cardiac_phase + additional_step
                    pixels_per_mask[cardiac_phase_ind,:,1] = current_cardiac_phase
                    pixels_per_mask[cardiac_phase_ind,:,2] = additional_step
                    self.log.node("AutoFocus working on cardiac phase " + str(current_cardiac_phase)+", stored in index: " + str(cardiac_phase_ind) + ", user picked phase was: " + str(cardiac_phase))
                    data_current_phase = data[current_cardiac_phase-1,:,:]
        
                    blurred_data = mask[current_cardiac_phase-1,:,:]*af.blur(data_current_phase.astype(np.complex64), rot_matrix.astype(np.complex64))
                    blurred_data_debug[cardiac_phase_ind,:,:,:] = blurred_data
                    
                    # modification starts here: Dan Zhu                  
                    if flow_encodes > 1:
                        blurred_data_flow = np.zeros([flow_encodes,nr_freq, matrix_size, matrix_size ], dtype=np.complex64 )
                        for flow_ind in range(flow_encodes):
                            data_current_phase_flow = data_flow[flow_ind,current_cardiac_phase-1,:,:]
                            blurred_data_flow[flow_ind,:,:,:]=mask[current_cardiac_phase-1,:,:]*af.blur(data_current_phase_flow.astype(np.complex64), rot_matrix.astype(np.complex64))
                            blurred_data_debug_flow[flow_ind,cardiac_phase_ind,:,:,:]=blurred_data_flow[flow_ind,:,:,:]
                    # modification ends here: Dan Zhu
                    
                    
                    xv, yv = np.meshgrid(np.linspace(-1, 1, int(2*nr_radii+1)), np.linspace(-1, 1, int(2*nr_radii+1)))
                    r=np.sqrt(xv**2+yv**2)
                    mask_circle=np.ones((int(2*nr_radii+1),int(2*nr_radii+1)))
                    mask_circle[np.where(abs(r)>1)]=0
                    
                    
                    kSpace = self.fft2D( np.array(blurred_data, dtype=np.complex64), dir=1)
                    Hann2_win = kaiser2D.window2(kSpace.shape[-2:], 20, 35)
                    kSpace=kSpace*Hann2_win
                    data_lowres=self.fft2D( np.array(kSpace, dtype=np.complex64), dir=0)
                    data_tmp=blurred_data/abs(data_lowres)*np.conj(data_lowres)
                    blurred_data=mask[current_cardiac_phase-1,:,:]*data_tmp                   
                    self.setData('Debug Data',blurred_data)
                    
                    
                    for point in range(nr_of_points):
                        px = points_phantom[point, 1]
                        py = points_phantom[point, 0]
                        
                        for freq in range(nr_freq):
                            current_data = mask_circle * blurred_data[freq,(px-int(nr_radii)):(px+int(nr_radii+1)),(py-int(nr_radii)):(py+int(nr_radii+1))]
                            intergral_sqrt_imaginary_area[cardiac_phase_ind,point,freq] = np.sum(np.sqrt( np.abs( np.imag(current_data))))
                
                        # choose deblurred image
                        #self.log.debug(intergral_sqrt_imaginary_area.argmin(2))
                        #self.log.debug(intergral_sqrt_imaginary_area.argmin(2)[cardiac_phase_ind,point])
                        deblur_freq[cardiac_phase_ind,point] = intergral_sqrt_imaginary_area.argmin(2)[cardiac_phase_ind,point]

                        if data_type == 0:
                            deblurred_data[cardiac_phase_ind,:,:] = blurred_data[deblur_freq[cardiac_phase_ind,point],:,:]
                            # modification starts here: Dan Zhu                  
                            if flow_encodes > 1:
                                for flow_ind in range(flow_encodes):
                                    deblurred_data_flow[flow_ind, cardiac_phase_ind,:,:] = blurred_data_flow[flow_ind,deblur_freq[cardiac_phase_ind,point],:,:]
                            # modification ends here: Dan Zhu
            
                # cardiac phases type "all": Use the frequency determined from user-defined location and phase for all cardiac phases to generate deblurred movie
                if cardiac_type == 0:
                    for phase in range(phases):
                        data_current_phase = data[phase,:,:]
                        rot_matrix_for_deblurring = rot_matrix[deblur_freq[0,0],:,:]
                        deblurred_data[phase,:,:] = mask[phase,:,:]*af.blur(data_current_phase.astype(np.complex64), rot_matrix_for_deblurring.astype(np.complex64))
                        
                        # modification starts here: Dan Zhu                  
                        if flow_encodes > 1:
                            for flow_ind in range(flow_encodes):
                                deblurred_data_flow[flow_ind, phase,:,:] = mask[phase,:,:]*af.blur(data_flow[flow_ind,phase,:,:].astype(np.complex64), rot_matrix_for_deblurring.astype(np.complex64))
                        # modification ends here: Dan Zhu
               
               
            else:
                nr_freq_when_not_deblurring = 1
                if cardiac_type == 0:
                    deblurred_data = np.zeros([phases, matrix_size, matrix_size ], dtype=np.complex64 )
                    blurred_data_debug = np.zeros([1, nr_freq_when_not_deblurring, matrix_size, matrix_size ], dtype=np.complex64 )
                    # modification starts here: Dan Zhu
                    if flow_encodes > 1:
                        deblurred_data_flow = np.zeros([flow_encodes, matrix_size, matrix_size ], dtype=np.complex64 )
                        blurred_data_debug_flow=np.zeros([flow_encodes,1, nr_freq_when_not_deblurring, matrix_size, matrix_size ], dtype=np.complex64 )
                    # modification ends here: Dan Zhu
                    cardiac_phases = 1
                else:
                    deblurred_data = np.zeros([cardiac_phases, matrix_size, matrix_size ], dtype=np.complex64 )
                    blurred_data_debug = np.zeros([cardiac_phases, nr_freq_when_not_deblurring, matrix_size, matrix_size ], dtype=np.complex64 )
                    # modification starts here: Dan Zhu
                    if flow_encodes > 1:
                        deblurred_data_flow = np.zeros([flow_encodes,cardiac_phases, matrix_size, matrix_size ], dtype=np.complex64 )
                        blurred_data_debug_flow=np.zeros([flow_encodes,cardiac_phases, nr_freq_when_not_deblurring, matrix_size, matrix_size ], dtype=np.complex64 )
                    # modification ends here: Dan Zhu
                deblur_freq = np.zeros([cardiac_phases, nr_of_points], dtype=np.int16)
                # number of pixels above FWHM in mask, and associated cardiac phase in cine and offset from chosen cardiac phase (cardiac_phase)
                pixels_per_mask = np.zeros([cardiac_phases, nr_of_points, 10])

                for cardiac_phase_ind in range(cardiac_phases):
                    additional_step = 1 - int(np.ceil(0.5*cardiac_phases)) + cardiac_phase_ind
                    current_cardiac_phase = cardiac_phase + additional_step
                    pixels_per_mask[cardiac_phase_ind,:,1] = current_cardiac_phase
                    pixels_per_mask[cardiac_phase_ind,:,2] = additional_step
                    #self.log.debug("AutoFocus working on cardiac phase " + str(current_cardiac_phase)+", stored in index: " + str(cardiac_phase_ind) + ", user picked phase was: " + str(cardiac_phase))
                    data_current_phase = data[current_cardiac_phase-1,:,:]*mask[current_cardiac_phase-1,...]
                    deblurred_data[cardiac_phase_ind,:,:] = data_current_phase
                    blurred_data_debug[cardiac_phase_ind,0,:,:] = data_current_phase
                    
                    # modification starts here: Dan Zhu
                    if flow_encodes > 1:
                        for flow_ind in range(flow_encodes):
                            data_current_phase_flow = data_flow[flow_ind,current_cardiac_phase-1,:,:]*mask[current_cardiac_phase-1,...]
                            deblurred_data_flow[flow_ind,cardiac_phase_ind,:,:] = data_current_phase_flow
                            blurred_data_debug_flow[flow_ind,cardiac_phase_ind,0,:,:]=data_current_phase_flow
                    # modification ends here: Dan Zhu
            
                # set f0_step for output even when deblurring was not performed
                f0_step = 0

            # determine FWHM area and Sharpness
            if FWHM_area:
                # use scipy library to detect connected areas and choose largest object only
                
                from scipy import ndimage
                from skimage import segmentation #if this package is missing install with: pip install -U scikit-image
                
                    

                nr_radii *= ZeroFill_factor
                
                if data_type == 0:
                    data_out = np.zeros([cardiac_phases, 7, 2*nr_radii,2*nr_radii], dtype = np.float32)
                else:
                    data_out = np.zeros([cardiac_phases, 7, nr_columns*2*nr_radii,nr_rows*2*nr_radii], dtype = np.float32)

                # Do different levels of thresholds in case the 50% threshold grows beyond the vessel
                FWHM_mask_array = np.zeros([cardiac_phases, 2 * nr_FWHM_steps, 2*nr_radii,2*nr_radii], dtype = np.float32)
                
                # sharpness values
                sharpness = np.zeros([cardiac_phases, nr_of_points])

                # SNR
                snr = np.zeros([cardiac_phases, nr_of_points])

                # instead of zero-filling each time, zero fill and filter images upfront, and then get those images from the loop.
                # Requires more memory but should be faster
                # for debugging purpose, implement with a "fast" switch
                # This is only faster for phantom analyis, therefore added the "fast" option only if there are more than 1 point
                fast = (nr_of_points > 1)
                if fast:
                    zero_filled_data = np.zeros([cardiac_phases, nr_freq, matrix_size * ZeroFill_factor, matrix_size * ZeroFill_factor ], dtype=np.complex64 )
                    zero_filled_edge = np.zeros([cardiac_phases, nr_freq, matrix_size * ZeroFill_factor, matrix_size * ZeroFill_factor ], dtype=np.float32 )
                    for cardiac_phase_ind in range(cardiac_phases):
                        for freq_index in range(nr_freq):
                            # get copy of data
                            tata = np.copy(blurred_data_debug[cardiac_phase_ind,freq_index,:,:])
                            # Zero fill deblurred image
                            out_dims = np.array([deblurred_data.shape[-1]*ZeroFill_factor,deblurred_data.shape[-1]*ZeroFill_factor], np.int64)
                            if has_filter:
                                zerofilled_data = af.filtered_zerofill(tata.astype(np.complex64), filter.astype(np.float32), out_dims)
                            else:
                                zerofilled_data = af.zerofill(tata.astype(np.complex64), out_dims)
                            zero_filled_data[cardiac_phase_ind, freq_index,:,:] = zerofilled_data

                            # Edge detection using Sobel filter for Sharpness measure. Filter before zero-filling
                            # Smooth and compute gradient
                            edge_data = np.abs(np.copy(blurred_data_debug[cardiac_phase_ind,freq_index,:,:]))
                            # from Mike, Gabri may want to change that again: edge_data = ndimage.filters.gaussian_filter(edge_data,.7) # !! may be a shallow copy !!
                            edge_data = np.sqrt(ndimage.filters.sobel(edge_data,axis=-1)**2 + ndimage.filters.sobel(edge_data,axis=-2)**2)
                            # from Mike, Gabri may want to change that again. Sobel filter gives a slope that is 4 times larger than I would expect.
                            edge_data /= 4.
                            # Zero fill..
                            zerofilled_edge = af.zerofill(edge_data.astype(np.complex64), out_dims)
                            zero_filled_edge[cardiac_phase_ind, freq_index,:,:] = np.abs(zerofilled_edge)



                for point in range(nr_of_points):
                    px = points_phantom[point, 1] * ZeroFill_factor
                    py = points_phantom[point, 0] * ZeroFill_factor
                    ##self.log.debug("px:"+str(px)+", py:"+str(py))

                    px_chosen_cardiac_phase = px
                    py_chosen_cardiac_phase = py
                    
                    info = "Areas measured with FWHM and Sharpness:\n"
                    info = info + "Cardiac phase    location    area [mm^2]    sharpness [%]     deblurr freq [Hz]     snr\n"
                    
                    for cardiac_phase_ind_to_be_rearranged in range(cardiac_phases):
                        self.log.node("working on point " + str(point) + " of " + str(nr_of_points) + " and cardiac phase " + str(cardiac_phase_ind_to_be_rearranged) + " of " + str(cardiac_phases))
                        if has_reacquired_data == 1:
                            cardiac_phase_ind = cardiac_phase_ind_to_be_rearranged
                        else:
                            # start at the chosen cardiac phase, then move up, and afterwards move down
                            cardiac_phase_ind = np.where(pixels_per_mask[:,point,2]==0)[0][0] + cardiac_phase_ind_to_be_rearranged
                            if cardiac_phase_ind == cardiac_phases:
                                cardiac_phase_ind = cardiac_phases - cardiac_phase_ind_to_be_rearranged -1
                                px = px_chosen_cardiac_phase
                                py = py_chosen_cardiac_phase
                                #self.log.debug("iter: " + str(iter) + ", px = "+str(px)+", py = "+str(py)+ "reset!")
                            elif cardiac_phase_ind > cardiac_phases:
                                cardiac_phase_ind = cardiac_phases - cardiac_phase_ind_to_be_rearranged -1
                            #self.log.debug("FWHM area determination working on for loop index: " + str(cardiac_phase_ind_to_be_rearranged) + ", which is array index: " + str(cardiac_phase_ind) + ", or actual cardiac phase: " + str(pixels_per_mask[cardiac_phase_ind,point,1]))
                    
                        if fast:
                            zerofilled_data = zero_filled_data[cardiac_phase_ind,deblur_freq[cardiac_phase_ind,point],:,:]
                        else:
                            # Zero fill deblurred image
                            tata = np.copy(blurred_data_debug[cardiac_phase_ind,deblur_freq[cardiac_phase_ind,point],:,:])
                            out_dims = np.array([deblurred_data.shape[-1]*ZeroFill_factor,deblurred_data.shape[-1]*ZeroFill_factor], np.int64)
                            if has_filter:
                                zerofilled_data = af.filtered_zerofill(tata.astype(np.complex64), filter.astype(np.float32), out_dims)
                            else:
                                zerofilled_data = af.zerofill(tata.astype(np.complex64), out_dims)
                        
                        # GE cine tool uses global minimum in image and not a local minimum
                        global_min_signal = np.abs(zerofilled_data).min()
                        
                        # In the first step find the pixel with maximum signal in a circular area around where the user clicked. Limit the circle to half of the chosen focus radius
                        # In step 2, threshold at various levels from 50% to 80% in steps of 5 between the local maximum and the global minimum.
                        # In step 3, the user will select the lowest threshold (largest area) that reflects the vessel.
                        # Last, determine (px, py) at center of mass of the 70% threshold, so that when moving from slice to slice we have the best possibility to track the vessel.
                        
                        # Step 1: Find maximum around user selected point. First create a circular mask.
                        # Cartesian ROI centered at point px,py of size radii
                        
                        radius = nr_radii // 2
                        c_data = np.abs(zerofilled_data[px-radius:px+radius, py-radius:py+radius])
                    
                        # create circular mask based on Cartesian ROI array
                        circle_mask = np.ones_like(c_data)
                        
                        # determine max within circle defined by user defined radius
                        for x in range(c_data.shape[-1]):
                            for y in range(c_data.shape[-2]):
                                if np.abs( (x-radius) + 1j*(y-radius) ) > radius:
                                    circle_mask[x,y] = 0

                        FWHM_mask = circle_mask * c_data
                        local_max_signal = FWHM_mask.max()

                        px_ROI_max,py_ROI_max = np.unravel_index(FWHM_mask.argmax(), FWHM_mask.shape)
                        # px_max and py_max in full FOV
                        px_max = int(px + px_ROI_max - radius)
                        py_max = int(py + py_ROI_max - radius)
                        
                        #self.log.debug("iter: " + str(iter) + ", px = "+str(px)+", py = "+str(py))

                        # 2nd Iteration: Repeat bassed on location of maximum signal.
                        # Additionally, determine Area's based on both local and global minimum, and the add 55, 60, 65, 70, 75% half-maximum areas.
                        # Display all and let the user decide which reflect the vessel.
                        c_data = np.abs(zerofilled_data[px_max-nr_radii:px_max+nr_radii, py_max-nr_radii:py_max+nr_radii])

                        radius = nr_radii
                        circle_mask = np.ones_like(c_data)
                        # determine max within circle defined by user defined radius
                        for x in range(c_data.shape[-1]):
                            for y in range(c_data.shape[-2]):
                                if np.abs( (x-radius) + 1j*(y-radius) ) > radius:
                                    circle_mask[x,y] = 0

                        actual_voxel_size = 1. * FOV / (matrix_size * ZeroFill_factor)
    
                        for FWHM_index in range(nr_FWHM_steps):
                            # determine FWHM and apply to masked data
                            # create mask based on Cartesian ROI array
                            FWHM_mask_current = circle_mask * c_data

                            half_max = (local_max_signal + global_min_signal) * 0.01 * (50. + (FWHM_index * FWHM_step_size) )
                            print("point: " + str(point) + "/" + str(nr_of_points) + ", FWHM_index: " + str(FWHM_index) + ", factor: " + str(0.01 * (50. + (FWHM_index * FWHM_step_size))) + ", half_max: " + str(half_max))
                            FWHM_mask_current = FWHM_mask_current >= half_max

                            # use scipy library to detect connected areas and choose largest object only
                            lbl, nlbl = ndimage.measurements.label(FWHM_mask_current)
                            # for phantom data trust that max_signal is within vessel, while for real vessel trust that user clicked in vessel
                            if data_type == 0:
                                my_label = lbl[radius,radius]
                            else:
                                max_indeces = np.unravel_index(np.argmax(c_data[radius//2:-radius//2,radius//2:-radius//2]), c_data[radius//2:-radius//2,radius//2:-radius//2].shape)
                                my_label = lbl[max_indeces[0]+(radius//2), max_indeces[1]+(radius//2)]
                            FWHM_mask_current = (lbl == my_label).astype(int)
                            FWHM_mask_current = ndimage.binary_fill_holes(FWHM_mask_current)
                            FWHM_mask_array[cardiac_phase_ind, FWHM_index,:,:] = FWHM_mask_current
                            
                            pixels_per_mask[ cardiac_phase_ind, point, 3 + FWHM_index ] = FWHM_mask_current.sum()

                            # multiply by voxel area
                            pixels_per_mask[ cardiac_phase_ind, point, 3 + FWHM_index ] *= actual_voxel_size * actual_voxel_size
                    
                            # boundaries
                            FWHM_mask_array[cardiac_phase_ind, nr_FWHM_steps + FWHM_index,:,:] = segmentation.find_boundaries(FWHM_mask_current)

                        pixels_per_mask[ cardiac_phase_ind, point, 0 ] = pixels_per_mask[ cardiac_phase_ind, point, 3 ]

                        # determine new pixel location with center of mass for start value of next cardiac phase based on 80% threshold when using in-vivo data
                        # for phantom data, only 50% threshold is determined
                        if data_type == 0:
                            location_threshold = 6
                        else:
                            location_threshold = 0
                        FWHM_mask = FWHM_mask_array[cardiac_phase_ind, location_threshold,:,:] == 1
                        px_ROI,py_ROI = ndimage.measurements.center_of_mass(FWHM_mask)

                        # px and py in full FOV
                        px = int(px_max + px_ROI - nr_radii)
                        py = int(py_max + py_ROI - nr_radii)


                        info = info + "                   " + str(int(pixels_per_mask[cardiac_phase_ind,point,1])) + "    " + str(int(1.*py_max/ZeroFill_factor)) + "/" + str(int(1.*px_max/ZeroFill_factor)) + "   " + str(pixels_per_mask[ cardiac_phase_ind, point, 0]) + "    " + str(global_min_signal) + "/" + str(local_max_signal)
                        
                        # create boundary of FWHM area for display
                        FWHM_mask = FWHM_mask_array[cardiac_phase_ind, 0,:,:] == 1
                        boundary = FWHM_mask_array[cardiac_phase_ind, nr_FWHM_steps,:,:]
                        
                        
                        # Zero-fill original image and crop
                        tata = np.copy(blurred_data_debug[cardiac_phase_ind,(nr_freq//2),:,:])
                        out_dims = np.array([blurred_data_debug.shape[-1]*ZeroFill_factor,blurred_data_debug.shape[-1]*ZeroFill_factor], np.int64)
                        if has_filter:
                            zerofilled_blurred_data = af.filtered_zerofill(tata.astype(np.complex64), filter.astype(np.float32), out_dims)
                        else:
                            zerofilled_blurred_data = af.zerofill(tata.astype(np.complex64), out_dims)
                        c_blurred_data = np.abs(zerofilled_blurred_data[px_max-nr_radii:px_max+nr_radii, py_max-nr_radii:py_max+nr_radii])


                        # determine SNR
                        # requires at least 2 cardiac phases which are subtracted for the noise determination. Use mean signal in segemented vessel, and standard deviation of noise in cropped ROI
                        # for noise, use original blurred data to have a better subtraction
                        # for noise crop data again with original px and py to ensure that the subtracted data haven't moved against each other.
                        if (cardiac_phases > 1):
                            if has_reacquired_data == 1:
                                if cardiac_phase_ind%2 == 0:
                                    # ref_data = c_data.copy()
                                    # for snr need to subtract images cropped at the same position
                                    px_max_chosen_cardiac_phase = px_max
                                    py_max_chosen_cardiac_phase = py_max
                                    ref_data = zerofilled_data[px_max_chosen_cardiac_phase-nr_radii:px_max_chosen_cardiac_phase+nr_radii, py_max_chosen_cardiac_phase-nr_radii:py_max_chosen_cardiac_phase+nr_radii]
                                    ref_data_noise = zerofilled_blurred_data[px_max_chosen_cardiac_phase-nr_radii:px_max_chosen_cardiac_phase+nr_radii, py_max_chosen_cardiac_phase-nr_radii:py_max_chosen_cardiac_phase+nr_radii]
                                    
                                    ref_mask = FWHM_mask.copy()
                                    # for debugging
                                    noise_data = ref_data.copy()
                                    snr[ cardiac_phase_ind, point ] = 0.
                                else:
                                    # this_data = np.abs(zerofilled_data[px_max_chosen_cardiac_phase-nr_radii:px_max_chosen_cardiac_phase+nr_radii, py_max_chosen_cardiac_phase-nr_radii:py_max_chosen_cardiac_phase+nr_radii])
                                    this_data = zerofilled_data[px_max_chosen_cardiac_phase-nr_radii:px_max_chosen_cardiac_phase+nr_radii, py_max_chosen_cardiac_phase-nr_radii:py_max_chosen_cardiac_phase+nr_radii]
                                    this_data_noise = zerofilled_blurred_data[px_max_chosen_cardiac_phase-nr_radii:px_max_chosen_cardiac_phase+nr_radii, py_max_chosen_cardiac_phase-nr_radii:py_max_chosen_cardiac_phase+nr_radii]
                                    noise_data = this_data_noise - ref_data_noise
                                    noise = 1. / np.sqrt(2) * np.std( np.real( noise_data ) )
                                    signal = 0.5 * np.mean( ( np.abs(this_data) + np.abs(ref_data) )[ref_mask] )
                                    snr[ cardiac_phase_ind, point ] = signal / noise
                                    #self.log.debug("noise: " + str(noise) + ", signal: " + str(signal) + ", and snr: " + str(snr[ cardiac_phase_ind, point]) )
                            else:
                                if cardiac_phase_ind_to_be_rearranged == 0:
                                    # ref_data = c_data.copy()
                                    # for snr need to subtract images cropped at the same position
                                    px_max_chosen_cardiac_phase = px_max
                                    py_max_chosen_cardiac_phase = py_max
                                    ref_data = zerofilled_data[px_max_chosen_cardiac_phase-nr_radii:px_max_chosen_cardiac_phase+nr_radii, py_max_chosen_cardiac_phase-nr_radii:py_max_chosen_cardiac_phase+nr_radii]
                                    ref_data_noise = zerofilled_blurred_data[px_max_chosen_cardiac_phase-nr_radii:px_max_chosen_cardiac_phase+nr_radii, py_max_chosen_cardiac_phase-nr_radii:py_max_chosen_cardiac_phase+nr_radii]
                                    
                                    ref_mask = FWHM_mask.copy()
                                    # for debugging
                                    noise_data = ref_data.copy()
                                    snr[ cardiac_phase_ind, point ] = 0.
                                else:
                                    # this_data = np.abs(zerofilled_data[px_max_chosen_cardiac_phase-nr_radii:px_max_chosen_cardiac_phase+nr_radii, py_max_chosen_cardiac_phase-nr_radii:py_max_chosen_cardiac_phase+nr_radii])
                                    this_data = zerofilled_data[px_max_chosen_cardiac_phase-nr_radii:px_max_chosen_cardiac_phase+nr_radii, py_max_chosen_cardiac_phase-nr_radii:py_max_chosen_cardiac_phase+nr_radii]
                                    this_data_noise = zerofilled_blurred_data[px_max_chosen_cardiac_phase-nr_radii:px_max_chosen_cardiac_phase+nr_radii, py_max_chosen_cardiac_phase-nr_radii:py_max_chosen_cardiac_phase+nr_radii]
                                    noise_data = this_data_noise - ref_data_noise
                                    noise = 1. / np.sqrt(2) * np.std( np.real( noise_data ) )
                                    signal = 0.5 * np.mean( ( np.abs(this_data) + np.abs(ref_data) )[ref_mask] )
                                    snr[ cardiac_phase_ind, point ] = signal / noise
                                    #self.log.debug("noise: " + str(noise) + ", signal: " + str(signal) + ", and snr: " + str(snr[ cardiac_phase_ind, point]) )
                        else:
                            noise_data = c_data.copy()
                        
                        # -----------------------------
                        # Vessel Sharpness Computation (GB - begin)
                        # -----------------------------
                        if fast:
                            edge_data = zero_filled_edge[cardiac_phase_ind,deblur_freq[cardiac_phase_ind,point],px_max-nr_radii:px_max+nr_radii, py_max-nr_radii:py_max+nr_radii]
                        else:
                            # Edge detection using Sobel filter for Sharpness measure. Filter before zero-filling
                            # Smooth and compute gradient
                            edge_data = np.abs(np.copy(blurred_data_debug[cardiac_phase_ind,deblur_freq[cardiac_phase_ind,point],:,:]))
                            # from Mike, Gabri may want to change that again: edge_data = ndimage.filters.gaussian_filter(edge_data,.7) # !! may be a shallow copy !!
                            edge_data = np.sqrt(ndimage.filters.sobel(edge_data,axis=-1)**2 + ndimage.filters.sobel(edge_data,axis=-2)**2)
                            # from Mike, Gabri may want to change that again. Sobel filter gives a slope that is 4 times larger than I would expect.
                            edge_data /= 4.
                            # Zero fill..
                            zerofilled_edge = af.zerofill(edge_data.astype(np.complex64), out_dims)
                            edge_data = zerofilled_edge[px_max-nr_radii:px_max+nr_radii, py_max-nr_radii:py_max+nr_radii]
                        
                        edge_data = edge_data / (local_max_signal-global_min_signal) * 100.
                        
                        # create radial spokes
                        nr_angles = 32
                        
                        # last iteration to find center is not performed anymore, center of vessel should be center of image
                        
                        spoke         = np.zeros([        1, nr_radii-1], dtype=np.float32) # dtype...
                        spokes_coords = np.zeros([nr_angles, nr_radii-1, 2], dtype=np.int16)
                        # compute max on spokes 
                        max_pks = np.zeros([nr_angles, 1], dtype=np.float32)
                        max_loc = np.zeros([nr_angles, 2], dtype=np.int16)
                        
                        # in some cases a spoke may not be withing the FWHM mask at all, in those cases don't count that spoke for the average value.
                        angle_counter = 0
                        for angle_ind in range(nr_angles):
                            # new, start spokes from max signal point, which is now at the center of the cropped data
                            # spokes_coords[angle_ind, :, 0] = np.int16(px_ROI_max + np.cos( 2. * np.pi * angle_ind / nr_angles) * np.arange(nr_radii-1))
                            # spokes_coords[angle_ind, :, 1] = np.int16(py_ROI_max + np.sin( 2. * np.pi * angle_ind / nr_angles) * np.arange(nr_radii-1))
                            # if the spokes should start at the maximum signal. This may be at the edge of the vessel
                            # spokes_coords[angle_ind, :, 0] = np.int16(nr_radii + np.cos( 2. * np.pi * angle_ind / nr_angles) * np.arange(nr_radii-1))
                            # spokes_coords[angle_ind, :, 1] = np.int16(nr_radii + np.sin( 2. * np.pi * angle_ind / nr_angles) * np.arange(nr_radii-1))
                            # set the center of spokes to the center of mass of the 70% max area:
                            spokes_coords[angle_ind, :, 0] = np.int16(px_ROI + np.cos( 2. * np.pi * angle_ind / nr_angles) * np.arange(nr_radii-1))
                            spokes_coords[angle_ind, :, 1] = np.int16(py_ROI + np.sin( 2. * np.pi * angle_ind / nr_angles) * np.arange(nr_radii-1))
                            #self.log.debug( "*** coords X *** "+str(spokes_coords[angle_ind, :, 0]))
                            #self.log.debug( "*** coords Y *** "+str(spokes_coords[angle_ind, :, 1]))

                            # correct coordinates if greater than dimensions
                            outbound_ind = np.nonzero(spokes_coords[angle_ind, :, 0] >= edge_data.shape[-1])[0]
                            if len(outbound_ind) > 0:
                                    #self.log.debug("*** GB *** coords X out of bound: "+str(outbound_ind))
                                    spokes_coords[angle_ind, outbound_ind, 0] = edge_data.shape[-1]-1
                            outbound_ind = np.nonzero(spokes_coords[angle_ind, :, 1] >= edge_data.shape[-2])[0]
                            if len(outbound_ind) > 0:
                                    #self.log.debug("*** GB *** coords Y out of bound: "+str(outbound_ind))
                                    spokes_coords[angle_ind, outbound_ind, 1] = edge_data.shape[-2]-1

                            FWHM_bound = np.nonzero(FWHM_mask[spokes_coords[angle_ind, :, 0] ,spokes_coords[angle_ind, :, 1]])[0]
                            spoke = edge_data[spokes_coords[angle_ind, FWHM_bound, 0] ,spokes_coords[angle_ind, FWHM_bound, 1]]        
                            #self.log.debug( "*** FWHM_bound ***"+str(FWHM_bound) )
                            if (FWHM_bound.size == 0):
                                self.setData('debug3', FWHM_mask)
                                max_loc[angle_ind, 0] = np.int16(px_ROI )
                                max_loc[angle_ind, 1] = np.int16(py_ROI )
                            else:
                                max_pks[angle_ind] = np.amax(spoke)
                                rad_loc = np.argmax(spoke)
                                #max_loc[angle_ind, 0] = np.int16(px_ROI + np.cos( 2. * np.pi * angle_ind / nr_angles) * rad_loc)
                                #max_loc[angle_ind, 1] = np.int16(py_ROI + np.sin( 2. * np.pi * angle_ind / nr_angles) * rad_loc)
                                max_loc[angle_ind, 0] = spokes_coords[angle_ind, rad_loc, 0]
                                max_loc[angle_ind, 1] = spokes_coords[angle_ind, rad_loc, 1]
                                angle_counter += 1


                        # Vessel sharpness as average value of maxima edge on the lumen border
                        sharpness[cardiac_phase_ind,point] = np.sum(max_pks) / angle_counter

                        # update info output with curr sharpness
                        info = info + "    "+str(sharpness[cardiac_phase_ind, point])+"  "+ str(-delta_f0 + ( deblur_freq[ cardiac_phase_ind, point] * f0_step )) + "  "+ str(snr[ cardiac_phase_ind, point ])+"\n"

                        # Create results images        
                        max_edge = edge_data.max()        
                        edge_spokes = np.copy(edge_data)
                        edge_spokes[spokes_coords[:,:,0].flatten('C'), spokes_coords[:,:,1].flatten('C')] = max_edge # should flatten rows first (C-like)
                        edge_border = 0.7 * np.copy(edge_data)
                        edge_border[max_loc[:,0].flatten(), max_loc[:,1].flatten()] = max_edge

                                                
                        # for debugging only
                        if data_type == 0:
                            data_out[cardiac_phase_ind,0,:,:] = c_data
                            data_out[cardiac_phase_ind,1,:,:] = c_blurred_data
                            data_out[cardiac_phase_ind,2,:,:] = FWHM_mask
                            data_out[cardiac_phase_ind,3,:,:] = boundary
                            data_out[cardiac_phase_ind,4,:,:] = edge_spokes
                            data_out[cardiac_phase_ind,5,:,:] = edge_border
                            data_out[cardiac_phase_ind,6,:,:] = noise_data
                        else:
                            if phantom == 4:
                                if point > 65:
                                    modified_point = point+1
                                else:
                                    modified_point = point

                                phantom_row = modified_point%11
                                phantom_col = modified_point//11
                            else:
                                phantom_row = point%nr_columns
                                phantom_col = point//nr_columns
                        
                            row_start = phantom_row * 2 * nr_radii
                            row_end = row_start + ( 2*nr_radii)
                            col_start = phantom_col * 2 * nr_radii
                            col_end = col_start + ( 2*nr_radii)

                            data_out[cardiac_phase_ind,0,row_start:row_end,col_start:col_end] = c_data
                            data_out[cardiac_phase_ind,1,row_start:row_end,col_start:col_end] = c_blurred_data
                            data_out[cardiac_phase_ind,2,row_start:row_end,col_start:col_end] = FWHM_mask
                            data_out[cardiac_phase_ind,3,row_start:row_end,col_start:col_end] = boundary
                            data_out[cardiac_phase_ind,4,row_start:row_end,col_start:col_end] = edge_spokes
                            data_out[cardiac_phase_ind,5,row_start:row_end,col_start:col_end] = edge_border
                            data_out[cardiac_phase_ind,6,row_start:row_end,col_start:col_end] = noise_data

                        # -----------------------------
                        # Vessel Sharpness Computation (GB - end)
                        # -----------------------------

                self.setData('FWHM_array', FWHM_mask_array)
                if data_type == 0:
                    self.setAttr('FWHM slider 50 55 60 65 70 75 80', visible=True)

                # order vessel area
                if data_type == 0:
                    phase_vessel_area_sharpness = np.zeros([cardiac_phases * sharpness.shape[-1], 14])
                    for phase_ind in range( cardiac_phases ):
                        for vessel_ind in range( sharpness.shape[-1] ):
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),0] = pixels_per_mask[ phase_ind, vessel_ind, 1]
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),1] = vessel_ind
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),2] = pixels_per_mask[ phase_ind, vessel_ind, 0]
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),3] = sharpness[ phase_ind, vessel_ind ]
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),4] = -delta_f0 + ( deblur_freq[ phase_ind,vessel_ind] * f0_step )
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),5] = snr[ phase_ind, vessel_ind ]
                            for FWHM_index in range(nr_FWHM_steps):
                                phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),6+FWHM_index] = pixels_per_mask[ phase_ind, vessel_ind, 3 + FWHM_index]
                    phase_vessel_area_sharpness[:,13] = self.getVal('FWHM slider 50 55 60 65 70 75 80')
                    
#                    FWHM_areas = np.zeros([cardiac_phases,2])
#                    for cardiac_phase_ind in range(cardiac_phases):
#                        FWHM_areas[ cardiac_phase_ind, 0 ] = pixels_per_mask[ cardiac_phase_ind, 0, 0]
#                        FWHM_areas[ cardiac_phase_ind, 1 ] = pixels_per_mask[ cardiac_phase_ind, 0, 1]
#                        ordered_sharpness = sharpness
                else:
                    phase_vessel_area_sharpness = np.zeros([cardiac_phases * sharpness.shape[-1], 7])
                    for phase_ind in range( cardiac_phases ):
                        for vessel_ind in range( sharpness.shape[-1] ):
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),0] = pixels_per_mask[ phase_ind, vessel_ind, 1]
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),1] = vessel_ind
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),2] = pixels_per_mask[ phase_ind, vessel_ind, 0]
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),3] = sharpness[ phase_ind, vessel_ind ]
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),4] = -delta_f0 + ( deblur_freq[ phase_ind,vessel_ind] * f0_step )
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),5] = snr[ phase_ind, vessel_ind ]
                            phase_vessel_area_sharpness[vessel_ind + (phase_ind  * sharpness.shape[-1]),6] = pixels_per_mask[ phase_ind, vessel_ind, 3]
            
                    # oder data for CHUV phantom
                    if phantom == 1:
                        nr_repeats = 5 # each size exists 5 times
                        nr_vessels = nr_of_points//nr_repeats
                        
                        size_order = [ [20,38,60,82,98],[10,40,57,80,104],[11,43,48,78,91],[12,34,49,83,89],[5,31,62,85,105],[13,30,66,81,101],[3,28,59,72,100],[22,41,52,68,93],[2,37,65,70,102],[1,35,61,74,92],[17,39,47,88,107],[15,33,64,87,97],[4,24,50,79,103],[9,26,58,71,99],[21,25,53,76,110],[8,23,51,69,90],[16,29,46,77,108],[7,32,54,73,106],[14,44,56,67,109],[18,42,45,75,94],[19,27,55,86,95],[6,36,63,84,96] ]
                        
                        if has_reacquired_data == 1:
                            phantom_area = np.zeros([2,nr_vessels, nr_repeats])
                            phantom_sharpness = np.zeros([2,nr_vessels, nr_repeats])
                            phantom_freq = np.zeros([2,nr_vessels, nr_repeats])
                            phantom_snr = np.zeros([2,nr_vessels, nr_repeats])
                            
                            for vessel in range(nr_vessels):
                                for occurance in range(nr_repeats):
                                    for phase_ind in range( cardiac_phases//2 ):
                                        phantom_area[ 0, vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + ((2*phase_ind  )  * nr_of_points), 2]
                                        phantom_area[ 1, vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + ((2*phase_ind+1)  * nr_of_points), 2]
                                    for phase_ind in range( cardiac_phases//2 ):
                                        phantom_sharpness[ 0, vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + ((2*phase_ind  )  * nr_of_points), 3]
                                        phantom_sharpness[ 1, vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + ((2*phase_ind+1)  * nr_of_points), 3]
                                    for phase_ind in range( cardiac_phases//2 ):
                                        phantom_freq[ 0, vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + ((2*phase_ind  )  * nr_of_points), 4]
                                        phantom_freq[ 1, vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + ((2*phase_ind+1)  * nr_of_points), 4]
                                    for phase_ind in range( cardiac_phases//2 ):
                                        phantom_snr[ 0, vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + ((2*phase_ind  )  * nr_of_points), 5]
                                        phantom_snr[ 1, vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + ((2*phase_ind+1)  * nr_of_points), 5]
                            phantom_sharpness /= cardiac_phases//2
                            phantom_area /= cardiac_phases//2
                            phantom_freq /= cardiac_phases//2
                            phantom_snr /= cardiac_phases//2
                        
                            self.setData('phantom_area', phantom_area)
                            self.setData('phantom_sharpness', phantom_sharpness)
                            self.setData('phantom_freq', phantom_freq)
                            self.setData('phantom_snr', phantom_snr)
                        else:
                            phantom_area = np.zeros([nr_vessels, nr_repeats])
                            phantom_sharpness = np.zeros([nr_vessels, nr_repeats])
                            phantom_freq = np.zeros([nr_vessels, nr_repeats])
                            phantom_snr = np.zeros([nr_vessels, nr_repeats])
                                    
                            for vessel in range(nr_vessels):
                                for occurance in range(nr_repeats):
                                    for phase_ind in range( cardiac_phases ):
                                        phantom_area[ vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + (phase_ind  * nr_of_points), 2]
                                    phantom_area[ vessel, occurance ] /= cardiac_phases
                                    for phase_ind in range( cardiac_phases ):
                                        phantom_sharpness[ vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + (phase_ind  * nr_of_points), 3]
                                    phantom_sharpness[ vessel, occurance ] /= cardiac_phases
                                    for phase_ind in range( cardiac_phases ):
                                        phantom_freq[ vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + (phase_ind  * nr_of_points), 4]
                                    phantom_freq[ vessel, occurance ] /= cardiac_phases
                                    if cardiac_phases > 1:
                                        for phase_ind in range( cardiac_phases ):
                                            phantom_snr[ vessel, occurance ] += phase_vessel_area_sharpness[(size_order[ vessel ][ occurance ] - 1) + (phase_ind  * nr_of_points), 5]
                                        phantom_snr[ vessel, occurance ] /= cardiac_phases - 1

                            self.setData('phantom_area', phantom_area)
                            self.setData('phantom_sharpness', phantom_sharpness)
                            self.setData('phantom_freq', phantom_freq)
                            self.setData('phantom_snr', phantom_snr)
    
    
                info = info + "\nFWHM              mean                 " + str(np.mean(pixels_per_mask[ :, :, 0 ])) + " and SD: " +  str(np.std    (pixels_per_mask[ :, :, 0 ])) + "\n"
                info = info +   "Sharpness        mean                 " + str(np.mean(sharpness[:, :])) + " and SD: " +  str(np.std(sharpness[:, :])) + "\n"

                self.setData('debug', data_out)
                
                #self.setData('debug 4', data_out3)
                self.setData('Cardiac Phase, vessel, FWHM area, sharpness, freq, snr', phase_vessel_area_sharpness)
                self.setAttr("Areas:", val=info, visible=True)
                
                                # cardiac phases type "all": Use the frequency determined from user-defined location and phase for all cardiac phases to generate deblurred movie
                if cardiac_type == 1:
                    sharpness_ave=np.mean(sharpness,axis=1)
                    phase_opt=np.argmax(sharpness_ave,axis=0)
                    for phase in range(phases):
                        rot_matrix_for_deblurring = rot_matrix[deblur_freq[phase_opt,0],:,:]                        
                        # modification starts here: Dan Zhu                  
                        if flow_encodes > 1:
                            for flow_ind in range(flow_encodes):
                                deblurred_data_flow[flow_ind, phase,:,:] = mask[phase,:,:]*af.blur(data_flow[flow_ind,phase,:,:].astype(np.complex64), rot_matrix_for_deblurring.astype(np.complex64))
                        # modification ends here: Dan Zhu

            else:
                self.setAttr("Areas:", val='', visible=False)

    
            # modification starts here: Dan Zhu
            self.setData('blurred array', np.squeeze(blurred_data_debug))
            self.setData('deblurred data', deblurred_data)
            if flow_encodes == 1:
                self.setData('blurred array flow', np.squeeze(blurred_data_debug))
                self.setData('deblurred data flow', deblurred_data)
            else:
                self.setData('blurred array flow', np.squeeze(blurred_data_debug_flow))
                self.setData('deblurred data flow', deblurred_data_flow)
            # modification ends here: Dan Zhu
            if doDeblur:
                self.setData('imaginary integral for deblurring', intergral_sqrt_imaginary_area)
                
            # end of compute
            self.setAttr('compute', val=False)
        elif compute:
            self.log.node("Point(s) needs to be selected in the image")

        debugging_data = self.getData('debug')
        deblurred_data_result = self.getData('deblurred data')
        FWHM_array_from_port = self.getData('FWHM_array')
        FWHM_slider_index = -1 + self.getVal('FWHM slider 50 55 60 65 70 75 80')
        if 'FWHM slider 50 55 60 65 70 75 80' in self.widgetEvents():
            self.log.node("FWHM Slider was changed. Update ouput data.")
            array_from_port = self.getData('Cardiac Phase, vessel, FWHM area, sharpness, freq, snr')
            if array_from_port is not None:
                array_from_port_copy = array_from_port.copy()
                array_from_port_copy[:,13] = self.getVal('FWHM slider 50 55 60 65 70 75 80')
                self.setData('Cardiac Phase, vessel, FWHM area, sharpness, freq, snr', array_from_port_copy)
        if debugging_data is not None:
            if data_type == 0:
                point = 0
                self.setAttr('results', visible=True)
                
                results_size_y = debugging_data.shape[-1]
                results_size_x = debugging_data.shape[-2]
                if doDeblur:
                    results_deblurring = self.getData('imaginary integral for deblurring')
                    results_nr_freq = results_deblurring.shape[-1]
                    results_output = np.zeros([results_size_x, 5 * results_size_y + 2*results_nr_freq ], dtype=np.float32)
                else:
                    results_output = np.zeros([results_size_x, 5 * results_size_y ], dtype=np.float32)
                
                if FWHM_area:
                    debugging_array = self.getData('Cardiac Phase, vessel, FWHM area, sharpness, freq, snr')
                    if debugging_array is not None:
                        min_cardiac_phase = np.min(debugging_array[:,0])
                        max_cardiac_phase = np.max(debugging_array[:,0])
                        for_cardiac_phase_ind = np.where(debugging_array[:,0]==cardiac_phase)[0]
                        if ( (cardiac_phase >= min_cardiac_phase) and (cardiac_phase <= max_cardiac_phase) ):
                            cardiac_phase_ind = int(cardiac_phase - min_cardiac_phase)
                            # For display: scale edge maps between [0 1] then scale according to max of deblurred image
                            print("*****************************************")
                            print(results_output.shape)
                            print("*****************************************")
                            results_output[:,0:int(results_size_y)] = debugging_data[cardiac_phase_ind,1,:,:]
                            results_output[:,int(results_size_y):2*int(results_size_y)] = debugging_data[cardiac_phase_ind,0,:,:]
                            results_output[:,2*int(results_size_y):3*int(results_size_y)] = 0.88 * debugging_data[cardiac_phase_ind,0,:,:] + 1.12 * debugging_data[cardiac_phase_ind,0,:,:].max() * FWHM_array_from_port[cardiac_phase_ind, nr_FWHM_steps + FWHM_slider_index,:,:]#debugging_data[cardiac_phase_ind,3,:,:]
                            results_output[:,3*int(results_size_y):4*int(results_size_y)] = (debugging_data[cardiac_phase_ind,4,:,:]) / debugging_data[cardiac_phase_ind,4,:,:].max() * 1.3 * debugging_data[cardiac_phase_ind,0,:,:].max() # edge_spokes
                            results_output[:,4*int(results_size_y):5*int(results_size_y)] = (debugging_data[cardiac_phase_ind,5,:,:]) / debugging_data[cardiac_phase_ind,5,:,:].max() * 1.3 *debugging_data[cardiac_phase_ind,0,:,:].max() # edge_border
                            max_value = results_output.max()
            
                            if (doDeblur and (data_type == 0) ):
                                results_deblurring_this_phase = results_deblurring[cardiac_phase_ind,0,:]
                                deblurring_max = results_deblurring_this_phase.max()
                                deblurring_min = results_deblurring_this_phase.min()
                            
                                for freq_ind in range(results_nr_freq):
                                    ix = (results_size_x-1) - int( (results_deblurring_this_phase[freq_ind] - deblurring_min) / (deblurring_max-deblurring_min) * (results_size_x-1) )
                                    iy = 5*results_size_y + 2*freq_ind
                                    results_output[ix:ix+2,iy:iy+2] = max_value
            else:
                if FWHM_area:
                    results_output = np.zeros_like(debugging_data[0,0,:,:])
                    debugging_array = self.getData('Cardiac Phase, vessel, FWHM area, sharpness, freq, snr')
                    if debugging_array is not None:
                        min_cardiac_phase = np.min(debugging_array[:,0])
                        max_cardiac_phase = np.max(debugging_array[:,0])
                        for_cardiac_phase_ind = np.where(debugging_array[:,0]==cardiac_phase)[0]
                        if ( (cardiac_phase >= min_cardiac_phase) and (cardiac_phase <= max_cardiac_phase) ):
                            cardiac_phase_ind = cardiac_phase - min_cardiac_phase
                            results_output = debugging_data[cardiac_phase_ind,0,:,:]



            image = np.abs(results_output)
                                       
            data_min = image.min()
            data_max = image.max()
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
            self.setAttr('results', val=image2)
            
        # Modification Starts Here: Dan Zhu: noted
        elif deblurred_data_result is not None:
            if deblurred_data_result.shape[-3] == data.shape[0]:
                self.setAttr('results', visible=True)
                
                results_deblurring = self.getData('imaginary integral for deblurring')
                results_nr_freq = results_deblurring.shape[-1]

                results_size = deblurred_data_result.shape[-1]
                results_size_quarter = int(0.25*results_size)
                results_size_half  = 2*results_size_quarter
                px = c_coords[1]
                py = c_coords[0]
                if px + results_size_quarter > results_size:
                    results_size_quarter = results_size - px - 1
                if py + results_size_quarter > results_size:
                    results_size_quarter = results_size - py - 1
                if px - results_size_quarter < 0:
                    results_size_quarter = px
                if py - results_size_quarter < 0:
                    results_size_quarter = py

                
                results_output = np.zeros([int(2*results_size_quarter), int(2*results_size_quarter) + 2*results_nr_freq ], dtype=np.float32)
                results_output[:,0:results_size_half ] = np.abs(deblurred_data_result[cardiac_phase-1, px-results_size_quarter:px+results_size_quarter, py-results_size_quarter:py+results_size_quarter ])

                max_value = results_output.max()

                results_deblurring_this_phase = results_deblurring[0,0,:]
                deblurring_max = results_deblurring_this_phase.max()
                deblurring_min = results_deblurring_this_phase.min()
                for freq_ind in range(results_nr_freq):
                    if (deblurring_max-deblurring_min) != 0:
                        ix = (results_size_half-1) - int( (results_deblurring_this_phase[freq_ind] - deblurring_min) / (deblurring_max-deblurring_min) * (results_size_half-1) )
                        iy = results_size_half + 2*freq_ind
                        results_output[ix:ix+2,iy:iy+2] = max_value

                image = np.abs(results_output)
    
                data_min = image.min()
                data_max = image.max()
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
                self.setAttr('results', val=image2)

        # Modification Ends Here: Dan Zhu

        return 0

    def execType(self):
        '''Could be GPI_THREAD, GPI_PROCESS, GPI_APPLOOP'''
        return gpi.GPI_APPLOOP #gpi.GPI_PROCESS  #Mike-debug: does it need to be an apploop for display widget?
    
    def fft2D(self, data, dir=0, out_dims_fft=[]):
        # data: np.complex64
        # dir: int (0 or 1) 0 = forward, 1 = inverse
        # outdims = [z,x,y]
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

        return corefft.fftw(data, outdims, **kwargs)

