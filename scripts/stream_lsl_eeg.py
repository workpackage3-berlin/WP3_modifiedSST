'''
(c) 2022,2023 Twente Medical Systems International B.V., Oldenzaal The Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

#######  #     #   #####   #
   #     ##   ##  #        
   #     # # # #  #        #
   #     #  #  #   #####   #
   #     #     #        #  #
   #     #     #        #  #
   #     #     #  #####    #

/**
 * @file ${example_stream_lsl.py} 
 * @brief This example shows the functionality to stream to LSL.
 *
 */


'''

from PySide2.QtWidgets import *
import sys
from os.path import join, dirname, realpath
scripts_dir = dirname(realpath(__file__)) # directory of this file
print(scripts_dir)
modules_dir = join(scripts_dir, '..') # directory with all modules
print(modules_dir)
measurements_dir = join(scripts_dir, '..\\measurements') # directory with all measurements
print(measurements_dir)
configs_dir = join(scripts_dir, '..\\TMSiSDK\\tmsi_resources') # directory with configurations
print(configs_dir)
sys.path.append(modules_dir)
import time

from TMSiSDK.tmsi_sdk import TMSiSDK, DeviceType, DeviceInterfaceType, DeviceState, ChannelType
from TMSiSDK.tmsi_errors.error import TMSiError, TMSiErrorCode, DeviceErrorLookupTable

from TMSiFileFormats.file_writer import FileWriter, FileFormat
from TMSiGui.gui import Gui
from TMSiPlotterHelpers.signal_plotter_helper import SignalPlotterHelper
from TMSiPlotterHelpers.impedance_plotter_helper import ImpedancePlotterHelper


try:
    # Execute a device discovery. This returns a list of device-objects for every discovered device.
    TMSiSDK().discover(dev_type = DeviceType.saga, dr_interface = DeviceInterfaceType.docked, ds_interface = DeviceInterfaceType.usb)
    discoveryList = TMSiSDK().get_device_list(DeviceType.saga)

    if (len(discoveryList) > 0):
        # Get the handle to the first discovered device.
        dev = discoveryList[0]
        
        # Open a connection to the SAGA-system
        dev.open()
        
        # Load the EEG channel set and configuration
        print("load EEG config")
        dev.import_configuration(join(configs_dir, "common_ref_config.xml"))
 
        # Check if there is already a plotter application in existence
        app = QApplication.instance()
        
        # Initialise the plotter application if there is no other plotter application
        if not app:
            app = QApplication(sys.argv)
            
        # Initialise the helper
        plotter_helper = ImpedancePlotterHelper(device=dev,
                                                 layout='head', 
                                                 file_storage = join(measurements_dir,"mSST_impedances"))
        # Define the GUI object and show it 
        gui = Gui(plotter_helper = plotter_helper)
         # Enter the event loop
        app.exec_()

        # Pause for a while to properly close the GUI after completion
        print('\n Wait for a bit while we close the plot... \n')
        print('\n Saving impedances values in measurements folder under "mSST_impedances-date-time"...')
        time.sleep(1)

        print("load recording config")
        dev.import_configuration(join(configs_dir, "common_ref_config.xml"))
        sf = dev.get_device_sampling_frequency()
        print(sf)


        # Initialise the lsl-stream
        stream = FileWriter(FileFormat.lsl, "SAGA")
        
        # Pass the device information to the LSL stream.
        stream.open(dev)
    
        # Check if there is already a plotter application in existence
        app = QApplication.instance()
        
        # Initialise the plotter application if there is no other plotter application
        if not app:
            app = QApplication(sys.argv)
            
        plotter_helper = SignalPlotterHelper(device=dev)
        # Define the GUI object and show it 
        gui = Gui(plotter_helper = plotter_helper)
         # Enter the event loop
        app.exec_()
        
        # Close the file writer after GUI termination
        stream.close()
        
        # Close the connection to the SAGA device
        dev.close()
    
except TMSiError as e:
    print(e)
    
        
finally:
    if 'dev' in locals():
        # Close the connection to the device when the device is opened
        if dev.get_device_state() == DeviceState.connected:
            dev.close()