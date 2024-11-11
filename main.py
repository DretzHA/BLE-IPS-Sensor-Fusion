import Calibration
import Movement

'''
IMPORTANT: The dataset file path needs to be specified correctly in the config.yaml file.
'''

'''
Run calibration to plot the RSSI curve and view the path-loss coefficient for each anchor.

Plot settings: Use the config.yaml file to determine whether to display no plots, only the first anchor plot, or all RSSI models.

The path-loss coefficient value must be manually assigned in config.yaml.
'''
Calibration.RSSI_model() 

'''
Run movements to plot and view the results for the estimated position provided by the IPS methods.

Plot settings: Use the config.yaml file to determine whether to display plots for the predicted path, average error, and Cumulative Distribution Function.
'''

Movement.run(case=1)
Movement.run(case=2)
Movement.run(case=3)
