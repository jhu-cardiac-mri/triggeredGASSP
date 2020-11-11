# Copyright (c) 2014, Dignity Health
# 
#     The GPI core node library is licensed under
# either the BSD 3-clause or the LGPL v. 3.
# 
#     Under either license, the following additional term applies:
# 
#         NO CLINICAL USE.  THE SOFTWARE IS NOT INTENDED FOR COMMERCIAL
# PURPOSES AND SHOULD BE USED ONLY FOR NON-COMMERCIAL RESEARCH PURPOSES.  THE
# SOFTWARE MAY NOT IN ANY EVENT BE USED FOR ANY CLINICAL OR DIAGNOSTIC
# PURPOSES.  YOU ACKNOWLEDGE AND AGREE THAT THE SOFTWARE IS NOT INTENDED FOR
# USE IN ANY HIGH RISK OR STRICT LIABILITY ACTIVITY, INCLUDING BUT NOT LIMITED
# TO LIFE SUPPORT OR EMERGENCY MEDICAL OPERATIONS OR USES.  LICENSOR MAKES NO
# WARRANTY AND HAS NOR LIABILITY ARISING FROM ANY USE OF THE SOFTWARE IN ANY
# HIGH RISK OR STRICT LIABILITY ACTIVITIES.
# 
#     If you elect to license the GPI core node library under the LGPL the
# following applies:
# 
#         This file is part of the GPI core node library.
# 
#         The GPI core node library is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version. GPI core node library is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
# 
#         You should have received a copy of the GNU Lesser General Public
# License along with the GPI core node library. If not, see
# <http://www.gnu.org/licenses/>.


# Author: Nick Zwart
# Date: 2012apr03

# Modified by Michael Schar
# Date: 20201030
# Allow reading dictionary, and set paths to the path of the read file

import os
import time
import gpi

class ExternalNode(gpi.NodeAPI):
    """Provides an interface to the python pickle (cPickle) module for
    de-serializing py-objects from a file.

    OUTPUT: Numpy array read from file

    WIDGETS:
    I/O Info - Gives info on data file and data type
    File Browser - button to launch file browser, and typein widget if the pathway is known.
    """
    def execType(self):
        # default executable type
        #return gpi.GPI_THREAD
        return gpi.GPI_PROCESS # this is the safest

    def initUI(self):

       # Widgets
        self.addWidget('TextBox', 'I/O Info:')
        self.addWidget('OpenFileBrowser', 'File Browser',
                button_title='Browse', caption='Open File',
                    filter='pickled (*.pickle);;all (*)')
        self.addWidget('TextEdit', 'File Name using out path from dictionary', val="extra_folder/output_filename", visible=False)

        # IO Ports
        self.addInPort('params_in', 'DICT', obligation=gpi.OPTIONAL)
        self.addOutPort('out','PASS')

        # store for later use
        self.URI = gpi.TranslateFileURI

    def validate(self):
        inparam = self.getData('params_in')
        use_Browser = False
        if ( inparam is not None):
            if ('out_path' in inparam):
                self.setAttr('File Browser', visible=False)
                self.setAttr('File Name using out path from dictionary', visible=True)
                file_name = self.getVal('File Name using out path from dictionary')
                fname = inparam['out_path'] + file_name
            else:
                use_Browser = True
        else:
            use_Browser = True
        if use_Browser:
            self.setAttr('File Browser', visible=True)
            self.setAttr('File Name using out path from dictionary', visible=False)
            fname = self.URI(self.getVal('File Browser'))
        self.setDetailLabel(fname)

    def compute(self):

        import os
        import time
        import pickle

        fname = self.getDetailLabel()

        # check that the path actually exists
        if not os.path.exists(fname):
            self.log.node("Path does not exist: "+str(fname))
            return 0

        # show some file stats
        fstats = os.stat(fname)
        # creation
        ctime = time.strftime('%d/%m/20%y', time.localtime(fstats.st_ctime))
        # mod time
        mtime = time.strftime('%d/%m/20%y', time.localtime(fstats.st_mtime))
        # access time
        atime = time.strftime('%d/%m/20%y', time.localtime(fstats.st_atime))
        # filesize
        fsize = fstats.st_size
        # user id
        uid = fstats.st_uid
        # group id
        gid = fstats.st_gid

        # read the data
        with open(fname, 'rb') as handle:
            out = pickle.load(handle)

        info = "created: "+str(ctime)+"\n" \
               "accessed: "+str(atime)+"\n" \
               "modified: "+str(mtime)+"\n" \
               "UID: "+str(uid)+"\n" \
               "GID: "+str(gid)+"\n" \
               "file size (bytes): "+str(fsize)+"\n" \
               "type: "+str(type(out))+"\n"
        self.setAttr('I/O Info:', val=info)

        # if this is a dictionary with parameters that include 'data_path" and 'out_path"
        # then modify those to reflect the current file location
        if type(out) == dict:
            if ( ('data_path' in out) and ('out_path' in out) ):

                file_path = os.path.dirname(os.path.abspath(fname))
                out_path = file_path + "/recon/"
                if not os.path.isdir( out_path):
                    os.mkdir( out_path )
                out['data_path'] = file_path
                out['out_path'] = out_path

        self.setData('out', out)

        return(0)
