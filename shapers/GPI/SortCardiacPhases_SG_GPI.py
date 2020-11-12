# Copyright 2015 Gabriele Bonanno
# Modified Dan Zhu, 2019, add phase contrast data compatibility
# Modified Michael Schar February, 2020  add shifted binning

import gpi
import numpy as np


class ExternalNode(gpi.NodeAPI):
    """ Sort data into cardiac phases
        Sort dynamic 2D data into cardiac phases based on RR interval information read from file
    """

    def initUI(self):
        # Widgets
        
        self.addWidget('PushButton', 'computenow', toggle=True, val=0)
        self.addWidget('SpinBox','MTX XY Reduction', min=5, max=100, val=100)
        self.addWidget('DoubleSpinBox','Eff MTX XY', min=5, val=240)
        self.addWidget('ExclusivePushButtons', 'cardiac triggers', buttons=('Self-Gating', 'ECG', 'ECG header', 'simulated'), val=0)
        self.addWidget('SpinBox', 'simulated heartrate [bpm]', immediate=True, val=75, visible=False)
        self.addWidget('SpinBox', 'simulated displacement [mm]', immediate=True, val=4, visible=False)
        self.addWidget('ExclusivePushButtons', 'cardiac phases',buttons=('Default', 'User Defined'),val=0)
        self.addWidget('SpinBox', '# of cardiac phases', immediate=True, val=40, visible=False)
        # Cardiac phase should be automated
        self.addWidget('SpinBox', 'sliding window factor [%]', immediate=True, val=50)
        self.addWidget('ExclusivePushButtons', 'Retrospective Window Optimization', buttons=('No', 'Best', 'Worst'), val=0)
        self.addWidget('SpinBox', 'Window Shift Steps', immediate=True, val=10, min=3, max=30)
        self.addWidget('PushButton', 'coords for BART', toggle=True, val=0)
        self.addWidget('PushButton', 'reduce data', toggle=True, val=0)
        self.addWidget('SpinBox', 'reduce data to [s]', immediate=True, min=1, max=20, val=10, visible=False)
        self.addWidget('PushButton', 'use self-consistent data', toggle=True, val=0)
        self.addWidget('SpinBox', 'use [s] of data', immediate=True, min=1, max=20, val=10, visible=False)
        self.addWidget('PushButton', 'use consistent heart beat durations', toggle=True, val=0)
        self.addWidget('SpinBox', 'use % of data', immediate=True, min=1, max=100, val=50, visible=False)
       
        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=[np.complex64, np.complex128], obligation=gpi.REQUIRED)
        self.addInPort('coords', 'NPYarray', dtype=[np.float64, np.float32], obligation=gpi.REQUIRED)
        self.addInPort('user_params', 'DICT', obligation=gpi.REQUIRED)
        self.addInPort('header', 'DICT', obligation=gpi.REQUIRED)       
 
        self.addInPort('R_top', 'NPYarray', dtype=[np.float64, np.float32], obligation=gpi.OPTIONAL)
        self.addInPort('RR_array', 'NPYarray', dtype=[np.float64, np.float32], obligation=gpi.OPTIONAL)
        self.addInPort('RR_number', 'NPYarray', dtype=[np.uint8], obligation=gpi.OPTIONAL)
#        self.addInPort('RR_number', 'NPYarray', dtype=[np.float64, np.float32], obligation=gpi.REQUIRED)
        # Gabri - RR_number is double in matlab but the GPI MatRead converts it in uint8 because doesn't find decimals prob
        self.addInPort('SG_data', 'NPYarray', dtype=[np.complex64, np.complex128], obligation=gpi.OPTIONAL)
        self.addInPort('Spiral Angles', 'NPYarray', dtype=[np.float32], obligation=gpi.OPTIONAL)


        self.addOutPort('sorted data', 'NPYarray')
        self.addOutPort('sorted coords', 'NPYarray')
        self.addOutPort('SDC wates', 'NPYarray') # binary weighting for the SDC computation to exclude some arms
        self.addOutPort('R_top_out', 'NPYarray')
        self.addOutPort('RR_array_out', 'NPYarray')
        self.addOutPort('RR_number_out', 'NPYarray')
        self.addOutPort('heart beats to be used')
        self.addOutPort('sorted SG_data')
        self.addOutPort('heart beats to be used based on RR')
        self.addOutPort('mean RMS for number of beats')
        self.addOutPort('debug')
        self.addOutPort('sorted Spiral Angles', 'NPYarray')
        self.addOutPort('HR Max Mean SD Spiral Gaps shift idx ms average_rr_without_outliers', 'NPYarray')
    
    def validate(self):
        param = self.getData('user_params')
        
        mtx_xy_reduction = self.getVal('MTX XY Reduction')
        if param is not None:
            if ( ('spFOVXY' in param) and ('spRESXY' in param) ):
                mtx_xy = 0.01 * mtx_xy_reduction * 1.25*float(param['spFOVXY'][0])/float(param['spRESXY'][0])
                self.setAttr('Eff MTX XY', quietval = mtx_xy)
    
        if self.getVal('cardiac phases') == 0:
            self.setAttr('# of cardiac phases', visible=False)
        else:
            self.setAttr('# of cardiac phases', visible=True)
    
        spiral_angles = self.getData('Spiral Angles')
        if spiral_angles is not None:
            self.setAttr('Retrospective Window Optimization', visible=True)
        else:
            self.setAttr('Retrospective Window Optimization', visible=False)
    
        if self.getVal('Retrospective Window Optimization') == 0:
            self.setAttr('Window Shift Steps', visible=False)
        else:
            self.setAttr('Window Shift Steps', visible=True)
    
        if self.getVal('reduce data') == 1:
            self.setAttr('reduce data to [s]', visible=True)
        else:
            self.setAttr('reduce data to [s]', visible=False)


        SG_data = self.getData('SG_data')
        if (( SG_data is not None) and (self.getVal('use self-consistent data') == 1)):
            self.setAttr('use [s] of data', visible=True)
        else:
            self.setAttr('use [s] of data', visible=False)
        
        if self.getVal('use consistent heart beat durations') == 1:
            self.setAttr('use % of data', visible=True)
        else:
            self.setAttr('use % of data', visible=False)
        
        if (self.getVal('cardiac triggers') == 3):
            self.setAttr('simulated heartrate [bpm]', visible=True)
            self.setAttr('simulated displacement [mm]', visible=True)
        else:
            self.setAttr('simulated heartrate [bpm]', visible=False)
            self.setAttr('simulated displacement [mm]', visible=False)

        return 0



    def compute(self):
        import numpy as np
        from scipy import linalg

        self.log.node("Data Sorting node running compute()")

        # GETTING WIDGET INFO
        mtx_xy_reduction = self.getVal('MTX XY Reduction')
        eff_mtx_xy = self.getVal('Eff MTX XY')
        mtx_xy = eff_mtx_xy * mtx_xy_reduction // 100
        # Nphases = self.getVal('cardiac phases')   # Dan Zhu Move downward
        slideWinFactor = self.getVal('sliding window factor [%]')
        slideWinOptimization = self.getVal('Retrospective Window Optimization')
        nr_window_shift_steps = 1
        if slideWinOptimization > 0:
            nr_window_shift_steps = self.getVal('Window Shift Steps')
        bartCoordsFlag = self.getVal('coords for BART')
        
        # GETTING PORT INFO
        data = self.getData('data').copy()
        coords = self.getData('coords').copy()
        inparam = self.getData('user_params')
        header = self.getData('header')
        spiral_angles = self.getData('Spiral Angles')   # Dan Zhu -- Spiral arm angles output
        
        if 'list' in header:
            hdr = header['list']
        else:
            hdr = header
        SG_data = self.getData('SG_data')
        if SG_data is not None:
            has_SG_data = True
        else:
            has_SG_data = False
        ECG_flag = self.getVal('cardiac triggers')
        
        # get total number of dynamics for following checks
        if 'spNUMDYN' in inparam:
            ndyn = int(float(inparam['spNUMDYN'][0]))
            self.log.node(" Reading ndyn " +str(ndyn))
        elif 'NUM_DYN' in inparam:
            ndyn = int(float(inparam['NUM_DYN']))
            self.log.node(" Reading ndyn " +str(ndyn))
        else:
            self.log.warn("*** !! num of dynamics not found in user_params! exiting..")        		
            return 1

        if ndyn != data.shape[-2]: #Dan Zhu
            self.log.warn("*** !! total number of dynamics from user_param and data size do not match! exiting..")
            return 1
        if coords.shape[0] != data.shape[-2]: #Dan Zhu
            self.log.warn("*** !! data and coords do not match in num of dynamics! exiting..")
            return 1

        is_flow_scan = False                                                                #Dan Zhu
        try:                                                                                #Dan Zhu
            if float(inparam['gn_flow_pc'][0]) == 1.0:                                      #Dan Zhu
                is_flow_scan = True                                                         #Dan Zhu
        except:                                                                             #Dan Zhu
            self.log.info("gn_flow_pc not in patch generated text file.")                   #Dan Zhu
        # independent of whether flow scan or not, need the following parameters:
        flow_dim = len(set(hdr['extr1']))                                                   #Dan Zhu
        FLow_Encoding=np.array(hdr['extr1'], dtype=np.int32)  # Just Channel 1              #Dan Zhu
        FLow_Encoding=FLow_Encoding[np.where((hdr['typ']=='STD') & (hdr['chan']=='0'))]     #Dan Zhu
        
        is_Trig_GA = False                                                                  #Dan Zhu
        try:                                                                                #Dan Zhu
            if float(inparam['ga_trig_rot'][0]) == 1.0:                                     #Dan Zhu
                is_Trig_GA = True                                                           #Dan Zhu
        except:                                                                             #Dan Zhu
            self.log.info("ga_trig_rot not in patch generated text file.")                  #Dan Zhu

        # MiS: For CSA scan with triggered GA rotation currently use flow label to indicate heartbeat toggle
        if (is_Trig_GA and not(is_flow_scan)):                                              #MiS
            flow_dim = 1                                                                    #MiS
            FLow_Encoding[:] = 0                                                            #MiS

        # MiS reduce data for quick reconstruction of low resolution scan based on widget 'MTX XY Reduction'
        if mtx_xy_reduction < 100:
            nr_points = data.shape[-1]
            magnitude_one_interleave = np.zeros(nr_points)
            for x in range(nr_points):
                magnitude_one_interleave[x] = np.sqrt( coords[0,x,0]**2 + coords[0,x,1]**2)
            k_max = np.max(magnitude_one_interleave)
            within_csm_width_radius = magnitude_one_interleave[:] < (0.01 * mtx_xy_reduction * k_max) 
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
                
            coords = coords[:,start_point:end_point,:] * 100 / mtx_xy_reduction
            if is_flow_scan:
                data = data[:,:,:,start_point:end_point]
            else:
                data = data[:,:,start_point:end_point]

        # triggers from HEADER file
        if ECG_flag == 2 and hdr is not None:
            
            number_of_arms = data.shape[-2]

            chan = np.array(hdr['chan'], dtype=np.int32)
            nr_channels = 1 + max(chan)

            rr_array_long = np.array(hdr['rr'], dtype=np.float32)
            rr_array_full = rr_array_long[nr_channels:(nr_channels*number_of_arms*flow_dim+1):nr_channels]       #Dan Zhu
            rr_array_full *= 0.001              #Dan Zhu

            r_top_array_long = np.array(hdr['rtop'], dtype=np.float32)
            r_top_array_full = r_top_array_long[nr_channels:(nr_channels*number_of_arms*flow_dim+1):nr_channels] #Dan Zhu
            r_top_array_full *= 0.001           #Dan Zhu

            # check size
            Nkt = rr_array_full.size            #Dan Zhu
            if not(Nkt == r_top_array_full.size): #Dan Zhu
                self.log.warn("*** !! rr_array and r_top_array from header have different size! exiting..")
                return 1
            elif not(Nkt == ndyn*flow_dim):     #Dan Zhu
                if not( (Nkt == ndyn) and is_Trig_GA):      #DanZ in TrigGA Nkt == ndyn, else Nkt == ndyn*flow_dim
                    self.log.warn("*** !! ndyn = " + str(ndyn) + " in header does not match size of data = " +str(Nkt) + "! exiting..")
                    return 1

            # compute rr number
            if rr_array_full.shape[-1] == r_top_array_full.shape[-1]: #Dan Zhu
              rr_number_full = np.zeros([Nkt], dtype=np.uint8) #Dan Zhu
              current_number = 0
              for i_dyn in range(Nkt-1): #Dan Zhu
                if r_top_array_full[i_dyn+1] - r_top_array_full[i_dyn] < 0: #Dan Zhu
                  current_number +=1
                rr_number_full[i_dyn+1] = current_number #Dan Zhu

            self.log.node("RR parameters read from HEADER!")

        elif ECG_flag == 3 and inparam is not None:
            simulated_heartrate = self.getVal('simulated heartrate [bpm]')
            simulated_displacement = self.getVal('simulated displacement [mm]')
            # scale to pixels
            if 'spRESXY' in inparam:
                voxel_size = float(inparam['spRESXY'][0])
            elif 'TRUE_RESXY' in inparam:
                voxel_size = float(inparam['TRUE_RESXY'])
            else:
                self.log.warn("*** !! voxel size could not be read from user_params! exiting..")
                return 1
            simulated_displacement /= voxel_size * 1000.

            if 'gnTR' in inparam:
                TR = float(inparam['gnTR'][0])
            elif 'TR' in inparam:
                TR = float(inparam['TR'])
            else:
                self.log.warn("*** !! TR could not be read from user_params! exiting..")
                return 1
            
            if 'spDWELLus' in inparam:
                dwell = float(inparam['spDWELLus'][0]) * 0.001
            elif 'DWELL_US' in inparam:
                dwell = float(inparam['DWELL_US']) * 0.001
            else:
                self.log.warn("*** !! Dwell time in us could not be read from user_params! exiting..")
                        

            nr_channels = data.shape[0]
            nr_samples = data.shape[-1]

            total_scan_duration = ndyn * TR * flow_dim #Dan Zhu
            
            # introducing new heartbeat simulation, using a SD of 50ms of the RR interval based on:
            # Nunan D. et al. A Quantitative Stematic Review of Normal Values for Short-Term Heart Rate Variability in Healthy Adults. Pace 33(11):1407-17, 2010
            new_method = True
            if new_method:
                SDNN = 50. #ms SDNN = standard deviation of normal-to-normal intervals  (means that the outlier heartbeats have been removed)
                average_heartbeat_duration = 60000. / simulated_heartrate
                np.random.seed( simulated_heartrate )
                
                # rr array:
                rr_array_full = np.ones([ndyn * flow_dim], dtype=np.float32)
                
                # compute r_top array
                r_top_array_full = np.ones([ndyn * flow_dim], dtype=np.float32)
                time = 0.
                heartbeat_duration = np.random.normal(average_heartbeat_duration,SDNN)
                # compute rr number
                rr_number_full = np.zeros([ndyn * flow_dim], dtype=np.uint8)
                current_number = 0
                for i_dyn in range(ndyn * flow_dim):
                    rr_array_full[ i_dyn ] = heartbeat_duration
                    r_top_array_full[ i_dyn ] = time
                    time += TR
                    if time > heartbeat_duration:
                        time -= heartbeat_duration
                        heartbeat_duration =  np.random.normal(average_heartbeat_duration,SDNN)
                    if i_dyn > 0:
                        if r_top_array_full[i_dyn] - r_top_array_full[i_dyn-1] < 0:
                            current_number +=1
                        rr_number_full[i_dyn] = current_number
            else:
                min_heartbeat_duration = 60000. / simulated_heartrate
                nr_heartbeats = total_scan_duration / min_heartbeat_duration

                # rr array: given by chosen heartrate, let the heartbeat duration increase by 1ms each beat.
                rr_array_full = np.ones([ndyn * flow_dim], dtype=np.float32) #Dan Zhu

                # compute r_top array
                r_top_array_full = np.ones([ndyn * flow_dim], dtype=np.float32) #Dan Zhu
                time = 0.
                heartbeat_duration = min_heartbeat_duration
                # compute rr number
                rr_number_full = np.zeros([ndyn * flow_dim], dtype=np.uint8) #Dan Zhu MiS-debug add * flow_dim????
                current_number = 0
                for i_dyn in range(ndyn * flow_dim):            #Dan Zhu MiS-debug add * flow_dim????
                    rr_array_full[ i_dyn ] = heartbeat_duration #Dan Zhu MiS-debug add _full????
                    r_top_array_full[ i_dyn ] = time            #Dan Zhu MiS-debug add _full????
                    time += TR
                    if time > heartbeat_duration:
                        time -= heartbeat_duration
                        heartbeat_duration += 3. * (i_dyn % 2)
                    if i_dyn > 0:
                        if r_top_array_full[i_dyn] - r_top_array_full[i_dyn-1] < 0:#Dan Zhu MiS-debug add _full????
                            current_number +=1
                        rr_number_full[i_dyn] = current_number #Dan Zhu

            # add linear phase in k-space to simulate motion when displacement amplitude is chosen > 0
            # systole at 2/7 heart beat
            systole = 2. / 7. * heartbeat_duration
            # systolic restperiod 40ms
            # diastole at 4.5/7 heart beat
            diastole = 4.5 / 7. * heartbeat_duration
            # diastolic restperiod 80ms
            
            if simulated_displacement > 0:
                displacement_array = np.zeros( [nr_samples * ndyn, 2] )
                
                time = 0.
                time_overhead = TR - (nr_samples * dwell)
                counter = 0
                for i_dyn in range(ndyn):
                  for i_flow in range(flow_dim):                    #Dan Zhu MiS-debug add ????
                    heartbeat_duration = rr_array_full[ (i_dyn * flow_dim) + i_flow ] #Dan Zhu
                    for i_sample in range(nr_samples):
                        if time > heartbeat_duration:
                            time -= heartbeat_duration
                        if time < systole:
                            displacement = simulated_displacement * (( 1 - np.cos(time / systole * np.pi) ) / 2.)
                        elif time < 40. + systole:
                            displacement = simulated_displacement
                        elif time < diastole:
                            displacement = simulated_displacement - (2. / 3. * simulated_displacement) * (( 1 - np.cos( (time-(systole + 40.)) / (diastole - systole - 40.) * np.pi) ) / 2.)
                        elif time < 80. + diastole:
                            displacement = 1. / 3. * simulated_displacement
                        else:
                            displacement = 1. / 3. * simulated_displacement - (1. / 3. * simulated_displacement) * (( 1 - np.cos( (time-(diastole + 80.)) / (heartbeat_duration - diastole - 80) * np.pi) ) / 2.)
                        displacement_array[ counter, 0 ] = displacement
                        displacement_array[ counter, 1 ] = time
                        counter += 1
                        # apply phase according to x-coordinate
                        if is_flow_scan:
                            data[:,i_flow,i_dyn,i_sample] *= np.exp(-1j * 2.0 * np.pi * (coords[i_dyn,i_sample,0] * displacement )) #Dan Zhu MiS-debug add i_flow
                        else:
                            print(data.shape)
                            data[:,i_dyn,i_sample] *= np.exp(-1j * 2.0 * np.pi * (coords[i_dyn,i_sample,0] * displacement )) #Dan Zhu MiS-debug add i_flow
                        time += dwell
                    time += time_overhead
                        
                self.setData('debug',displacement_array)
            Nkt = ndyn


        else: # triggers from MAT file

            r_top_array_full = self.getData('R_top')    #Dan Zhu
            rr_array_full = self.getData('RR_array')    #Dan Zhu
            rr_number_full = self.getData('RR_number')  #Dan Zhu

            if r_top_array_full is None:  #Dan Zhu
                self.log.warn("*** !! r_top_array not found! exiting..")
                return 1
            if rr_array_full is None:  #Dan Zhu
                self.log.warn("*** !! rr_array not found! exiting..")
                return 1
            if rr_number_full is None: #Dan Zhu
                self.log.warn("*** !! rr_number not found! exiting..")
                return 1

            # check size
            Nkt = rr_array_full.size  #Dan Zhu
            self.log.node("Nkt "+str(Nkt)) 
            if not(Nkt == r_top_array_full.size == rr_number_full.size):  #Dan Zhu
                self.log.warn("*** !! rr_array, rr_number and r_top_array have different size! exiting..")
                return 1
            if ((not is_Trig_GA) and (Nkt != ndyn * flow_dim)):  #Dan Zhu
                self.log.warn("*** !! ndyn in mat file does not match user params! exiting..")        		
                return 1

            self.log.node("RR parameters read from MAT files!")

        # crop data in case of reduce data when no SG data is available
        if (self.getVal('reduce data') == 1):
            if 'gnTR' in inparam:
                TR = float(inparam['gnTR'][0])
            else:
                self.log.warn("*** !! TR could not be read from user_params! exiting..")
                return 1
            max_dyn = int( 1000. * self.getVal('reduce data to [s]') / TR )
            if max_dyn < (ndyn * flow_dim):                 #Dan Zhu MiS-debug
                ndyn = max_dyn
                Nkt = ndyn
                if is_flow_scan:                            #Dan Zhu MiS-debug
                    data = data[:,:,0:max_dyn//flow_dim,...]#Dan Zhu MiS-debug
                else:                                       #Dan Zhu MiS-debug
                    data = data[:,0:max_dyn,...]            #Dan Zhu MiS-debug
                coords = coords[0:max_dyn,...]
                r_top_array_full = r_top_array_full[0:max_dyn]   #Dan Zhu
                rr_array_full = rr_array_full[0:max_dyn]         #Dan Zhu
                rr_number_full = rr_number_full[0:max_dyn]       #Dan Zhu
                if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
                    spiral_angles = spiral_angles[0:max_dyn]       #Dan Zhu

        # give out rr parameters before computing
        self.setData('R_top_out',     r_top_array_full)     #Dan Zhu
        self.setData('RR_array_out',  rr_array_full)        #Dan Zhu
        self.setData('RR_number_out', rr_number_full)       #Dan Zhu
        if self.getVal('cardiac phases') == 0:
            _, rr_change_idx=np.unique(rr_number_full,return_index=True)
            if 'gnTR' in inparam:
                TR = float(inparam['gnTR'][0])
            elif 'TR' in inparam:
                TR = float(inparam['TR'])
            else:
                self.log.warn("*** !! TR could not be read from user_params! exiting..")
                return 1
            Nphases=int(round(np.mean(rr_array_full[rr_change_idx[1:]]) / TR *1000))
            #hide this instead, otherwise the node restarts the network after finishing
            #self.setAttr('# of cardiac phases',val=Nphases)
        else:
            Nphases = self.getVal('# of cardiac phases')   # Dan Zhu Move here

        if self.getVal('computenow'):
          # MiS add retrospective window shift optimization
          u_rr_number_full, u_idx = np.unique(rr_number_full, return_index=True)
          u_rr_array_full = rr_array_full[u_idx]
          RealHR = 60 / np.mean(u_rr_array_full) # calculate HR
          max_gaps_shift = np.zeros([nr_window_shift_steps], dtype=np.float32)
          mean_gaps_shift = np.zeros([nr_window_shift_steps], dtype=np.float32)
          std_gaps_shift = np.zeros([nr_window_shift_steps], dtype=np.float32)
          current_max_gap = 0.
          current_min_gap = 360.
          for shift_idx in np.arange(nr_window_shift_steps):
            # Dan Zhu (begin)
            if is_flow_scan:
                max_arms_all=0
                # allocate sorted data with dims [Nch Nphases Nmax_arms Nrdtpoints]
                sorted_data   = np.zeros(shape=(data.shape[0],flow_dim, Nphases, data.shape[-2], data.shape[-1]),dtype=data.dtype)
                if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
                    sorted_spiral_angles  = np.zeros(shape=(flow_dim, Nphases,data.shape[-2]),dtype=spiral_angles.dtype)   # Dan Zhu -- Spiral arm angles output
                    max_spiral_gaps  = np.zeros(shape=(flow_dim, Nphases),dtype=spiral_angles.dtype)   # Dan Zhu -- Spiral arm angles output
            
                # allocate sorted coords with dims   [Nphases Nmax_arms Nrdtpoints 2]
                if bartCoordsFlag:
                    # For BART: assign default value to .5 or .45 for the coords that are not used after sorting in phases. 
                    sorted_coords = np.ones(shape=(flow_dim,Nphases, coords.shape[-3], coords.shape[-2], coords.shape[-1]),dtype=coords.dtype) * np.max(coords)
                else:
                    sorted_coords = np.zeros(shape=(flow_dim,Nphases, coords.shape[-3], coords.shape[-2], coords.shape[-1]),dtype=coords.dtype)
                                    # allocate weights for sorted coords with dims [Nphases Nmax_arms Nrdtpoints]
                wates = np.zeros(shape=(flow_dim,Nphases, coords.shape[-3], coords.shape[-2]),dtype=np.float64)
                          
            for flow_ind in range(flow_dim):
                r_top_array = r_top_array_full[FLow_Encoding==flow_ind]
                rr_array = rr_array_full[FLow_Encoding==flow_ind]
                rr_number =rr_number_full[FLow_Encoding==flow_ind]
                if (is_Trig_GA and is_flow_scan):  # DanZ TrigGA: rr_number 0,2,4-> 0,1,2  rr_number 1,3,5-> 0,1,2
                    rr_number=np.int32(rr_number/2)
                    
                Nkt = rr_array.size	
                if is_flow_scan:
                    if is_Trig_GA:   # DanZ TrigGA:
                        data_red=data[:,flow_ind,FLow_Encoding==flow_ind,...]
                    else:
                        data_red=data[:,flow_ind,...]
                else:
                    data_red=data
                
                if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
                    if is_Trig_GA:   # DanZ TrigGA:
                        spiral_angles_red=spiral_angles[FLow_Encoding==flow_ind]   # Dan Zhu -- Spiral arm angles output
                    else:   # Dan Zhu -- Spiral arm angles output
                        spiral_angles_red=spiral_angles   # Dan Zhu -- Spiral arm angles output
                
                if is_Trig_GA:   # DanZ TrigGA:
                    coords_red=coords[FLow_Encoding==flow_ind,...]
                else:
                    coords_red=coords

            # Dan Zhu (end)
                # ----------------------------------------------------
                # remove first RR interval)
                # ----------------------------------------------------
            
                # remove very first dynamics labeled with zero (incomplete RR) and first RR
                # last RR is now included as its RR_array value is set to RR_array_mean
                zeroRR_idx = np.nonzero(np.array(rr_number == 0))
                firstRR_idx = np.nonzero(np.array(rr_number == 1)) 
    #            lastRR_idx = np.nonzero(np.array(rr_number == rr_number[-1])) 

                self.log.node("*** zero RR arms to rmv " +str(zeroRR_idx[0]))
                if (coords.shape[-1] == 3):
                    armsToRemove_idx = zeroRR_idx[0]
                else:
                    self.log.node("*** first RR arms to rmv " +str(firstRR_idx[0]))
    #            self.log.node("*** last  RR arms to rmv " +str(lastRR_idx[0]))

    #            armsToRemove_idx = np.concatenate((zeroRR_idx[0],firstRR_idx[0],lastRR_idx[0]),axis=0)
                    armsToRemove_idx = np.concatenate((zeroRR_idx[0],firstRR_idx[0]),axis=0)

                if len(armsToRemove_idx)>0:
                    # remove stuff
                    rr_array    = np.delete(rr_array,armsToRemove_idx,axis=0)
                    rr_number   = np.delete(rr_number,armsToRemove_idx,axis=0)
                    r_top_array = np.delete(r_top_array,armsToRemove_idx,axis=0)
                    data_red        = np.delete(data_red,armsToRemove_idx,axis=-2)   #Dan Zhu
                    if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
                        spiral_angles_red=np.delete(spiral_angles_red,armsToRemove_idx,axis=0)   # Dan Zhu -- Spiral arm angles output
                    coords_red      = np.delete(coords_red,armsToRemove_idx,axis=0)  #Dan Zhu
                    if has_SG_data:
                        SG_data = np.delete(SG_data,armsToRemove_idx,axis=1)
                    
                    # update tot numb of arms
                    Nkt = Nkt - len(armsToRemove_idx)
                    self.log.node("*** removed individual RRs - arms tot: " +str(len(armsToRemove_idx))) 
                    self.log.node("*** updated Nkt: " +str(Nkt)) 

                # average RR-interval duration
                u_rr_number, u_idx = np.unique(rr_number, return_index=True)
                u_rr_array = rr_array[u_idx]
                median_rr = np.median(u_rr_array)
                average_rr = np.mean(u_rr_array)
                std_rr = np.std(u_rr_array)

                # ----------------------------------------------------
                # Find consistent heart beat durations and remove outliers
                # ----------------------------------------------------
                if self.getVal('use consistent heart beat durations') == 1:
                    self.log.node("Find consistent heart beat durations and remove outliers")
                    
                    u_rr_number, u_idx = np.unique(rr_number, return_index=True)
                    u_rr_array = rr_array[u_idx]
                    self.log.node("rr interval indeces: "+str(u_rr_number))
                    self.log.node("rr interval duration: "+str(u_rr_array))
                    self.log.node("rr interval median: "+str(median_rr))
                    self.log.node("rr interval mean: "+str(average_rr))
                    
                    nr_beats = np.unique(rr_number).size
                    # reduced rr_number array
                    r_rr_number = np.zeros_like(rr_number).flatten()
                    current_beat = 0
                    for arm in range(Nkt-1):
                        if rr_number[arm] != rr_number[arm+1]:
                            current_beat += 1
                        r_rr_number[arm+1] = current_beat

                    # Least-squares difference between all heart beats (along the cardiac phases)
                    hb_L2_array = np.zeros([nr_beats,nr_beats])
                    for line1 in range(nr_beats-1):
                        ind1 = line1 + 1
                        for line2 in range(ind1):
                            ind2 = line2
                            hb_L2_array[ind1,ind2] = np.sqrt(( u_rr_array[ind1] - u_rr_array[ind2] )**2)

                    # result is a lower triangular matrix. transpose and add to fill the matrix. The diagonal is still zero (subtracting the same beat leads to zero difference)
                    hb_L2_array  += hb_L2_array.transpose()

                    # sort the errors, to have the lowest error first for each beat
                    sorted_hb_L2_array = np.sort(hb_L2_array, axis=1)
                    # rember the indexes pre sorting
                    hb_L2_sort_ind = np.argsort(hb_L2_array, axis=1)

                    # for each beat (column), add the error of the previous beats. This will allow to find the beat with the lowest error for any given sized set of beats.
                    for line2 in range(nr_beats-1):
                        ind2 = line2+1
                        sorted_hb_L2_array[:,ind2] = sorted_hb_L2_array[:,line2] + sorted_hb_L2_array[:,ind2]

                    # Array of beats to use for any given duration of data to be used
                    hb_heart_beats_to_use = np.zeros([nr_beats,nr_beats])
                    # First row shows the number of heart beats to be used
                    hb_heart_beats_to_use[0,:] = np.arange(nr_beats)
                    mean_rms_for_nr_of_beats = np.zeros(nr_beats)
                    for beats in range(nr_beats-1):
                        min_ind = np.argmin(sorted_hb_L2_array[:,beats+1])
                        mean_rms_for_nr_of_beats[beats+1] = sorted_hb_L2_array[min_ind, beats+1] / beats
                        print("for " + str(beats+2) +" beats, use heart beat " + str(min_ind) + " with these beats:")
                        for which in range(beats+2):
                            print(hb_L2_sort_ind[min_ind, which])
                            hb_heart_beats_to_use[beats+1,which]=hb_L2_sort_ind[min_ind, which]
                    
                    self.setData('heart beats to be used based on RR', hb_heart_beats_to_use)
                    self.setData('mean RMS for number of beats', mean_rms_for_nr_of_beats)
                    
                    # reduce data to either the user defined percentage
                    reduced_data_percentage = self.getVal('use % of data')
                    nr_beats_to_use = np.int(np.ceil(0.01 * reduced_data_percentage * nr_beats))
                        
                    beats_used = hb_heart_beats_to_use[ nr_beats_to_use, :nr_beats_to_use ]
                    self.log.debug("For the best " + str(reduced_data_percentage) + "%, the following " + str(nr_beats_to_use) + " beats are used: " + str(beats_used))

                    # create idx array to remove entries from data and coords
                    idx_to_keep = np.logical_or( r_rr_number == beats_used[0], r_rr_number == beats_used[1] )
                    for beat in range(nr_beats_to_use-2):
                        idx_to_keep = np.logical_or( idx_to_keep, r_rr_number == beats_used[beat + 2])
                    idx_to_remove = np.nonzero(np.logical_not(idx_to_keep))
                    
                    if len(idx_to_remove[0])>0:
                        # remove stuff
                        rr_array    = np.delete(rr_array,idx_to_remove[0],axis=0)
                        r_top_array = np.delete(r_top_array,idx_to_remove[0],axis=0)
                        rr_number = np.delete(rr_number,idx_to_remove[0],axis=0)
                        data_red        = np.delete(data_red,idx_to_remove[0],axis=-2)   #Dan Zhu
                        if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
                            spiral_angles_red=np.delete(spiral_angles_red,idx_to_remove[0],axis=0)   # Dan Zhu -- Spiral arm angles output
                        coords_red      = np.delete(coords_red,idx_to_remove[0],axis=0)  #Dan Zhu

                        # update tot numb of arms
                        Nkt = Nkt - len(idx_to_remove[0])
                        # update average/std RR-interval duration
                        u_rr_number, u_idx = np.unique(rr_number, return_index=True)
                        u_rr_array = rr_array[u_idx]
                        average_rr = np.mean(u_rr_array)
                        std_rr = np.std(u_rr_array)
                        self.log.node("removed worst heart beats total: " +str(len(idx_to_remove[0])))
                        self.log.node("removed worst heart beats number: " +str(idx_to_remove[0]))
                        self.log.node("updated Nkt: " +str(Nkt))



                # ----------------------------------------------------
                # outlier RR-interval removal (arrythmia?)
                # ----------------------------------------------------
                self.log.node("outlier RR-interval removal (arrythmia?)")

                u_rr_number, u_idx = np.unique(rr_number, return_index=True)
                u_rr_array = rr_array[u_idx]
                self.log.node("rr interval indeces: "+str(u_rr_number))
                self.log.node("rr interval duration: "+str(u_rr_array))
                self.log.node("rr interval median: "+str(median_rr))
                self.log.node("rr interval mean: "+str(average_rr))

                outlierFactor = 0.3

                rrToRemove_idx = np.fabs(u_rr_array - median_rr) > outlierFactor*np.fabs(median_rr) 
                rrToRemove_idx = np.nonzero(rrToRemove_idx)

                self.log.node("outlier rule is +/- "+str(outlierFactor*np.fabs(median_rr))+" from median")
                self.log.node("identified outlier rr number:   " +str(u_rr_number[rrToRemove_idx[0]]))
                self.log.node("identified outlier rr duration: " +str(u_rr_array[rrToRemove_idx[0]]))
                  
                # apply same rule on all rr_array
                armsToRemove_idx = np.fabs(rr_array - median_rr) > outlierFactor*np.fabs(median_rr) 
                armsToRemove_idx = np.nonzero(armsToRemove_idx)

                
                if len(armsToRemove_idx[0])>0:
                    # remove stuff
                    rr_array    = np.delete(rr_array,armsToRemove_idx[0],axis=0)
                    r_top_array = np.delete(r_top_array,armsToRemove_idx[0],axis=0)
                    #MiS-debug why diferent? rr_number = np.delete(rr_number,idx_to_remove[0],axis=0)
                    rr_number = np.delete(rr_number,armsToRemove_idx[0],axis=0)
                    data_red        = np.delete(data_red,armsToRemove_idx[0],axis=-2)    #Dan Zhu
                    if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
                        spiral_angles_red=np.delete(spiral_angles_red,armsToRemove_idx[0],axis=0)   # Dan Zhu -- Spiral arm angles output
                    coords_red      = np.delete(coords_red,armsToRemove_idx[0],axis=0)   #Dan Zhu
                    if has_SG_data:
                        SG_data   = np.delete(SG_data,armsToRemove_idx[0],axis=1)
                        rr_number = np.delete(rr_number,armsToRemove_idx[0],axis=0)
                    
                    

                    # update tot numb of arms
                    Nkt = Nkt - len(armsToRemove_idx[0])
                    # update average/std RR-interval duration
                    u_rr_number, u_idx = np.unique(rr_number, return_index=True)
                    u_rr_array = rr_array[u_idx]
                    average_rr = np.mean(u_rr_array)
                    std_rr = np.std(u_rr_array)
                    self.log.node("removed outlier arms tot: " +str(len(armsToRemove_idx[0])))
                    self.log.node("removed outlier arm n.: " +str(armsToRemove_idx[0]))
                    self.log.node("updated Nkt: " +str(Nkt))
                #self.setData('debug',r_top_array)

                #self.setData('debug',relative_time)

                # target time based on average RR-interval and desired number of retrospective phases (Nphases+1 to get actual Nphases intervals)
                target_time = np.array(list(range(Nphases+1)), dtype=np.float32) / Nphases
                        
                # duration of target cardiac phase
                phase_dur = target_time[1]

                # Sorting SG data, like standard retrospective sorting to have 1 data point for each cardiac phase from each heart beat
                if has_SG_data:
                    SG_phases = 20
                    SG_slideWinFactor = 50
                    SG_target_time = np.array(list(range(SG_phases+1)), dtype=np.float32) / SG_phases
                    SG_phase_dur = SG_target_time[1] * (1 + 2*SG_slideWinFactor/100)
                    SG_target_time = np.array(list(range(SG_phases+1)), dtype=np.float32) / SG_phases
                    
                    # -------------------------------------------------------------------------------------------
                    # Compute relative time (rt) for each dynamic with respect to normalized cardiac phases (0:1)
                    # -------------------------------------------------------------------------------------------

                    # stretch each RR-interval non-linearly to align with the average RR-interval
                    # !!! there might be a division by zero because the first arm in RR has time from trigger = 0 (r_top_array)
                    SG_relative_time = np.ndarray(shape=(Nkt,1), dtype=np.float32)

                    if ECG_flag == 0:            
                        self.log.node('*** Self-Gating trigger data')
                        for ky in range(Nkt): # self-gating data: don't do non-linear stretching
                            s = rr_array[ky] / average_rr
                            SG_relative_time[ky] = float(r_top_array[ky]) / rr_array[ky]

                    else:
                        self.log.node('*** ECG trigger data: applying non-linear stretching')
                        for ky in range(Nkt): # ECG data: do non-linear stretching
                            s = rr_array[ky] / average_rr
                            t = float(r_top_array[ky]) / rr_array[ky]

                            if s <= 1.:
                                    SG_relative_time[ky] = (s * t) / ((s * t) - t + 1)
                            else:
                                    SG_relative_time[ky] = (2 * s * t) / ( 1 + np.sqrt(1 + 4 * s * t * (s - 1)) )
                        
                        
                    
                    nr_beats = np.unique(rr_number).size
                    sorted_SG_data = np.zeros([SG_phases, nr_beats], dtype=np.float32)
                    # reduced rr_number array
                    r_rr_number = np.zeros_like(rr_number).flatten()
                    current_beat = 0
                    for arm in range(Nkt-1):
                        if rr_number[arm] != rr_number[arm+1]:
                            current_beat += 1
                        r_rr_number[arm+1] = current_beat


                    # rms the channels
                    rms_SG_data = np.sqrt(np.sum(np.abs(SG_data)**2, axis=0))
                    # sum 10 samples at end of SG_data
                    rms_SG_data = np.sum(rms_SG_data[...,-15:-5], axis=-1).flatten()

                    # modified code from readPhilipsExports.py in gpi/jhu/ based on Philips retrospective binning.
                    for current_beat in range(nr_beats):
                        # relative times for this beat
                        relative_time_this_beat = SG_relative_time[r_rr_number == current_beat]
                        rms_SG_data_this_beat = rms_SG_data[r_rr_number == current_beat]
                        
                        # loop over interpolated phase, and find the measured data within a given window for that phase
                        for card in range(SG_phases):
                            window = SG_phase_dur
                            
                            # d is the difference between actual time - target time
                            d = np.array(relative_time_this_beat,dtype = np.float32) - SG_target_time[card]
                            # data acquired just before a new rtop is measured can be used for the first heart rate
                            dd = np.array(relative_time_this_beat,dtype = np.float32) - SG_target_time[card] - 1
                            # check if there is data in the target window, if not double the window size
                            while np.logical_or( (np.abs(d)<window), (np.abs(dd)<window) ).sum() == 0:
                                window *= 2
                            in_window = np.logical_or( (np.abs(d)<window), (np.abs(dd)<window) )
                            # p is weighting function within window
                            p = (1. + np.cos(np.pi * d / window))/2
                            p_sum = 0.
                            number_rtops_in_window = 0
                            rtops_in_window_index = []
                            for rtop in range(len(d)):
                                if in_window[rtop]:
                                    p_sum += p[rtop]
                                    rtops_in_window_index.append( rtop )
                                    number_rtops_in_window += 1
                            # loop over acquired profiles within window of current retrospective phase
                            for rtop in range(number_rtops_in_window):
                                sorted_SG_data[card, current_beat] += rms_SG_data_this_beat[int(rtops_in_window_index[rtop])] * p[int(rtops_in_window_index[rtop])] / p_sum

                    # Least-squares difference between all heart beats (along the cardiac phases)
                    L2_array = np.zeros([nr_beats,nr_beats])
                    for line1 in range(nr_beats-1):
                        ind1 = line1 + 1
                        for line2 in range(ind1):
                            ind2 = line2
                            L2_array[ind1,ind2] = np.sum(( sorted_SG_data[:,ind1] - sorted_SG_data[:,ind2] )**2)

                    # result is a lower triangular matrix. transpose and add to fill the matrix. The diagonal is still zero (subtracting the same beat leads to zero difference)
                    L2_array  += L2_array.transpose()

                    # sort the errors, to have the lowest error first for each beat
                    sorted_L2_array = np.sort(L2_array, axis=1)
                    # rember the indexes pre sorting
                    L2_sort_ind = np.argsort(L2_array, axis=1)

                    # for each beat (column), add the error of the previous beats. This will allow to find the beat with the lowest error for any given sized set of beats.
                    for line2 in range(nr_beats-1):
                        ind2 = line2+1
                        sorted_L2_array[:,ind2] = sorted_L2_array[:,line2] + sorted_L2_array[:,ind2]

                    # Array of beats to use for any given duration of data to be used
                    heart_beats_to_use = np.zeros([nr_beats,nr_beats])
                    # First row shows the duration for the number of heat beats to be used, if average RR-interval is available
                    heart_beats_to_use[0,:] = average_rr * np.arange(nr_beats)
                    for beats in range(nr_beats-1):
                        min_ind = np.argmin(sorted_L2_array[:,beats+1])
                        #print("for " + str(beats+2) +" beats, use heart beat " + str(min_ind) + " with these beats:")
                        for which in range(beats+2):
                            #print(L2_sort_ind[min_ind, which])
                            heart_beats_to_use[beats+1,which]=L2_sort_ind[min_ind, which]

                    self.setData('heart beats to be used', heart_beats_to_use)

                    self.setData('R_top_out',     r_top_array)
                    self.setData('RR_array_out',  rr_array)
                    self.setData('RR_number_out', rr_number)


                    # reduce data to either the user defined duration, or a hard-coded percentage
                    # get the TR first
                    if 'gnTR' in inparam:
                        TR = float(inparam['gnTR'][0])
                    else:
                        self.log.warn("*** !! TR could not be read from user_params! exiting..")
                        return 1
                    if (self.getVal('use self-consistent data') == 1):
                        reduced_data_duration = self.getVal('use [s] of data')
                    else:
                        hard_coded_percentage = 0.7
                        reduced_data_duration = hard_coded_percentage* Nkt * TR * 0.001
                    nr_beats_to_use = np.sum( heart_beats_to_use[0,:] < reduced_data_duration )
                        
                    beats_used = heart_beats_to_use[ nr_beats_to_use, :nr_beats_to_use ]
                    self.log.debug("For the best " + str(reduced_data_duration) + "s, the following " + str(nr_beats_to_use) + " beats are used: " + str(beats_used))

                    # create idx array to remove entries from data and coords
                    idx_to_keep = np.logical_or( r_rr_number == beats_used[0], r_rr_number == beats_used[1] )
                    for beat in range(nr_beats_to_use-2):
                        idx_to_keep = np.logical_or( idx_to_keep, r_rr_number == beats_used[beat + 2])
                    idx_to_remove = np.nonzero(np.logical_not(idx_to_keep))
                    
                    if len(idx_to_remove[0])>0:
                        # remove stuff
                        rr_array    = np.delete(rr_array,idx_to_remove[0],axis=0)
                        r_top_array = np.delete(r_top_array,idx_to_remove[0],axis=0)
                        rr_number = np.delete(rr_number,idx_to_remove[0],axis=0)
                        data_red        = np.delete(data_red,idx_to_remove[0],axis=-2)  #Dan Zhu
                        if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
                            spiral_angles_red=np.delete(spiral_angles_red,idx_to_remove[0],axis=0)   # Dan Zhu -- Spiral arm angles output
                        coords_red      = np.delete(coords_red,idx_to_remove[0],axis=0) #Dan Zhu

                        # update tot numb of arms
                        Nkt = Nkt - len(idx_to_remove[0])
                        # update average/std RR-interval duration
                        u_rr_number, u_idx = np.unique(rr_number, return_index=True)
                        u_rr_array = rr_array[u_idx]
                        average_rr = np.mean(u_rr_array)
                        std_rr = np.std(u_rr_array)
                        self.log.node("removed worst heart beats total: " +str(len(idx_to_remove[0])))
                        self.log.node("removed worst heart beats number: " +str(idx_to_remove[0]))
                        self.log.node("updated Nkt: " +str(Nkt))

    #self.setData('R_top_out',     r_top_array)
    #                self.setData('RR_array_out',  rr_array)
    #                self.setData('RR_number_out', rr_number)
                
                
                
                # -------------------------------------------------------------------------------------------
                # Compute relative time (rt) for each dynamic with respect to normalized cardiac phases (0:1)
                # -------------------------------------------------------------------------------------------

                # stretch each RR-interval non-linearly to align with the average RR-interval
                # !!! there might be a division by zero because the first arm in RR has time from trigger = 0 (r_top_array)
                relative_time = np.ndarray(shape=(Nkt,1), dtype=np.float32)

                if ECG_flag == 0:            
                    self.log.node('*** Self-Gating trigger data')
                    for ky in range(Nkt): # self-gating data: don't do non-linear stretching
                        s = rr_array[ky] / average_rr
                        relative_time[ky] = float(r_top_array[ky]) / rr_array[ky]

                else:
                    self.log.node('*** ECG trigger data: applying non-linear stretching')
                    for ky in range(Nkt): # ECG data: do non-linear stretching
                        s = rr_array[ky] / average_rr
                        t = float(r_top_array[ky]) / rr_array[ky]

                        if s <= 1.:
                                relative_time[ky] = (s * t) / ((s * t) - t + 1)
                        else:
                                relative_time[ky] = (2 * s * t) / ( 1 + np.sqrt(1 + 4 * s * t * (s - 1)) )

                # apply sliding window
                phase_dur *= (1 + 2*slideWinFactor/100) # (same factor is applied on both sides - corresponds to overlap between adjacent wins)

                self.log.node("target time size " +str(target_time.size))
                self.log.node("target time " +str(target_time))
                self.log.node("phase dur " +str(phase_dur))
                self.log.node("relative time size " +str(relative_time.size))
                self.log.node("data shape " +str(data.shape))
                self.log.node("coords shape " +str(coords.shape))
                    
                # MiS retrospective window shift optimization
                target_time += 1. / nr_window_shift_steps * phase_dur * ( (nr_window_shift_steps/2.) - shift_idx )

                # --------------------------------------
                # sort data in cardiac phases using arrays
                # --------------------------------------
                max_arms = 0 # max number of arms per cardiac phase
                arms_phase_list = []
                # assignment of each arm to a certain phase based on its relative_time with respect to the respective target_time
                # considering intervals of [-phase_dur/2 +phase_dur/2] centered on target_time[idx], \
                # thus the intervals at target_time[0]=0 and target_time[Nphases]=1 contain fewer data and will be merged
                # this implementation should be better for sliding window (makes sure bound intervals have similar n. of arms)
                for phase_idx in range(0,Nphases+1):
                    curr_ky_idx = np.nonzero(np.logical_and(relative_time >= target_time[phase_idx]-phase_dur/2,
                                    relative_time <  target_time[phase_idx]+phase_dur/2))
                    # np.nonzero gives a tuplet [array(),array()] for the 2 elements as indeces
                    arms_phase_list.append(curr_ky_idx[0])
                    if len(curr_ky_idx[0])>max_arms:
                        max_arms = len(curr_ky_idx[0])

                # merge last with first intervals
                self.log.node("len(arms_phase_list)= "+str(len(arms_phase_list))+" [0]= "+str(len(arms_phase_list[0]))+" [Nphases]= "+str(len(arms_phase_list[Nphases])))
                arms_phase_list[0] = np.concatenate((arms_phase_list[0],arms_phase_list[Nphases]))
                arms_phase_list = np.delete(arms_phase_list,Nphases,axis=0)
                self.log.node("len(arms_phase_list)= "+str(len(arms_phase_list))+" [0]= "+str(len(arms_phase_list[0]))+" [Nphases-1]= "+str(len(arms_phase_list[Nphases-1])))
                # update max arms
                if len(arms_phase_list[0])>max_arms:
                        max_arms = len(arms_phase_list[0])

                self.log.node("max_arms " +str(max_arms))

                # allocate sorted data with dims [Nch Nphases Nmax_arms Nrdtpoints] (already done for flow scans above. Could be nicer.)
                if is_flow_scan==0: #Dan Zhu
                    sorted_data   = np.zeros(shape=(data.shape[0], Nphases, max_arms, data.shape[-1]),dtype=data.dtype)
                    if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
                        sorted_spiral_angles   = np.zeros(shape=(Nphases, max_arms),dtype=spiral_angles.dtype)
                        max_spiral_gaps  = np.zeros(shape=(flow_dim, Nphases),dtype=spiral_angles.dtype)   # Dan Zhu -- Spiral arm angles output
                    # allocate sorted coords with dims   [Nphases Nmax_arms Nrdtpoints 2]
                    if bartCoordsFlag:
                        # For BART: assign default value to .5 or .45 for the coords that are not used after sorting in phases. 
                        sorted_coords = np.ones(shape=(Nphases, max_arms, coords.shape[-2], coords.shape[-1]),dtype=coords.dtype) * np.max(coords)
                    else:
                        sorted_coords = np.zeros(shape=(Nphases, max_arms, coords.shape[-2], coords.shape[-1]),dtype=coords.dtype)
                    # allocate weights for sorted coords with dims [Nphases Nmax_arms Nrdtpoints]
                    wates = np.zeros(shape=(Nphases, max_arms, coords.shape[-2]),dtype=np.float64)
                
                self.log.node("data shape " +str(sorted_data.shape))
                self.log.node("coords shape " +str(sorted_coords.shape))
                self.log.node("wates shape " +str(wates.shape))

                # populate sorted data and coords
                for phase_idx in range(Nphases):
                    if is_flow_scan:                                                                                                        #Dan Zhu
                        max_arms_all = max(max_arms_all,max_arms)                                                                           #Dan Zhu
                        print('*********************')
                        print(phase_idx,':       ',r_top_array[arms_phase_list[phase_idx]]/rr_array[arms_phase_list[phase_idx]])
                        sorted_data[:,flow_ind,phase_idx,0:len(arms_phase_list[phase_idx]),:]   = data_red[:,arms_phase_list[phase_idx],:]  #Dan Zhu
                        if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
                            sorted_spiral_angles[flow_ind,phase_idx,0:len(arms_phase_list[phase_idx])]   = spiral_angles_red[arms_phase_list[phase_idx]]  #Dan Zhu
                            angles=np.sort(spiral_angles_red[arms_phase_list[phase_idx]],kind='mergesort', order=None)   # Dan Zhu -- Spiral arm angles output
                            gaps=[*(angles[1:]-angles[0:-1]),angles[0]-angles[-1]+360.] #2.0*np.pi]   # Dan Zhu -- Spiral arm angles output
                            max_spiral_gaps[flow_ind,phase_idx]=np.max(np.abs(gaps))   # Dan Zhu -- Spiral arm angles output
                        sorted_coords[flow_ind,phase_idx,0:len(arms_phase_list[phase_idx]),:,:] = coords_red[arms_phase_list[phase_idx],:,:]#Dan Zhu
                        wates[flow_ind,phase_idx,0:len(arms_phase_list[phase_idx]),:] = 1.0                                                 #Dan Zhu
                    else:                                                                                                                   #Dan Zhu
                        sorted_data[:,phase_idx,0:len(arms_phase_list[phase_idx]),:]   = data_red[:,arms_phase_list[phase_idx],:]           #Dan Zhu data_red
                        if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
                            sorted_spiral_angles[phase_idx,0:len(arms_phase_list[phase_idx])]   = spiral_angles_red[arms_phase_list[phase_idx]]  #Dan Zhu
                            angles=np.sort(spiral_angles_red[arms_phase_list[phase_idx]],kind='mergesort', order=None)   # Dan Zhu -- Spiral arm angles output
                            gaps=[*(angles[1:]-angles[0:-1]),angles[0]-angles[-1]+360.] #2.0*np.pi]   # Dan Zhu -- Spiral arm angles output
                            max_spiral_gaps[flow_ind,phase_idx]=np.max(np.abs(gaps))   # Dan Zhu -- Spiral arm angles output
                        sorted_coords[phase_idx,0:len(arms_phase_list[phase_idx]),:,:] = coords_red[arms_phase_list[phase_idx],:,:]         #Dan Zhu coords_red
                        wates[phase_idx,0:len(arms_phase_list[phase_idx]),:] = 1.0


        #            print data[:,arms_phase_list[phase_idx],:].shape
        #            print range(len(arms_phase_list[phase_idx]))
        #            print "**** Gabri **** cardiac phase "+str(phase_idx)+" relative time vec "+str(np.transpose(relative_time[arms_phase_list[phase_idx]]))
                    self.log.node("phase "+str(phase_idx)+" arms idx vec "+str(arms_phase_list[phase_idx]))
        #            print sorted_data[:,phase_idx,0:len(arms_phase_list[phase_idx]),:].shape
        #            print wates[phase_idx,:,:]
        #            print "**** Gabri **** cardiac phase "+str(phase_idx)+" num of arms "+str(len(arms_phase_list[phase_idx]))

        
            if is_flow_scan:                                            #Dan Zhu
                sorted_data=sorted_data[...,0:max_arms_all,:]           #Dan Zhu
                if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
                    sorted_spiral_angles=sorted_spiral_angles[...,0:max_arms_all]           #Dan Zhu
                sorted_coords=sorted_coords[...,0:max_arms_all,:,:]     #Dan Zhu
                wates=wates[...,0:max_arms_all,:,:]                     #Dan Zhu
            
            if spiral_angles is not None:
                max_gaps_shift[shift_idx]=np.max(max_spiral_gaps)
                mean_gaps_shift[shift_idx]=np.mean(max_spiral_gaps)
                std_gaps_shift[shift_idx]=np.std(max_spiral_gaps)
                if (slideWinOptimization == 1): #best
                    if ( mean_gaps_shift[shift_idx] < current_min_gap ):
                        current_min_gap = mean_gaps_shift[shift_idx]
                        output_sorted_data = sorted_data.copy()
                        output_sorted_coords = sorted_coords.copy()
                        output_wates = wates.copy()
                        output_sorted_spiral_angles = sorted_spiral_angles.copy()
                elif (slideWinOptimization == 2): #worst
                    if ( mean_gaps_shift[shift_idx] > current_max_gap ):
                        current_max_gap = mean_gaps_shift[shift_idx]
                        output_sorted_data = sorted_data.copy()
                        output_sorted_coords = sorted_coords.copy()
                        output_wates = wates.copy()
                        output_sorted_spiral_angles = sorted_spiral_angles.copy()
                else:
                    output_sorted_data = sorted_data.copy()
                    output_sorted_coords = sorted_coords.copy()
                    output_wates = wates.copy()
                    output_sorted_spiral_angles = sorted_spiral_angles.copy()
            else:
                output_sorted_data = sorted_data.copy()
                output_sorted_coords = sorted_coords.copy()
                output_wates = wates.copy()
                if spiral_angles is not None:
                    output_sorted_spiral_angles = sorted_spiral_angles.copy()
            
          best_shift_idx = 0
          if (slideWinOptimization == 1): #best
            best_shift_idx = np.argmin(mean_gaps_shift)
          elif (slideWinOptimization == 2): #worst
            best_shift_idx = np.argmax(mean_gaps_shift)
          # select best shift
          max_gaps = max_gaps_shift[best_shift_idx]
          mean_gaps = mean_gaps_shift[best_shift_idx]
          std_gaps = std_gaps_shift[best_shift_idx]
            
          # SETTING PORT INFO
          self.setData('sorted data', output_sorted_data)
          self.setData('sorted coords', output_sorted_coords)
          self.setData('SDC wates', output_wates)
          if has_SG_data:
              self.setData('sorted SG_data', sorted_SG_data)
          if spiral_angles is not None:   # Dan Zhu -- Spiral arm angles output
              self.setData('sorted Spiral Angles', output_sorted_spiral_angles)
              shift_ms = 1000. * average_rr / nr_window_shift_steps * phase_dur * ( (nr_window_shift_steps/2.) - best_shift_idx )
              self.setData('HR Max Mean SD Spiral Gaps shift idx ms average_rr_without_outliers', np.array([RealHR,max_gaps,mean_gaps, std_gaps, best_shift_idx, shift_ms, 1000. * average_rr]))

        #MiS self.setAttr('computenow', val=0)

        return 0

    def execType(self):
        '''Could be GPI_THREAD, GPI_PROCESS, GPI_APPLOOP'''
        return gpi.GPI_PROCESS
