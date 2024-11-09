import yaml
import numpy as np
import Calibration
import Movement



'''
Run calibration to plot the RSSI curve and find the Path-loss coefficent to each anchor
Plot graphics: use the config.yaml to determine whether to display no plots, only the first plot, or all RSSI models.
The value of Path-loss coefficent has to be manually assigned in confing.yaml
'''
# Calibration.RSSI_model() 


Movement.run(case=1)
Movement.run(case=2)
Movement.run(case=3)
