# BLE-IPS-Sensor-Fusion

This repository was made available to perform sensor fusion using Angle-RSSI Fusion Localization (ARFL) with the dataset provided by Girolami et. al.

Details about ARFL can be found in the article:
A. Fabris, O. K. Rayel, J.L. Rebelatto, G. L. Moritz, and R. D. Souza, "AoA And RSSI-based BLE Indoor Positioning System wtih Kalman Filter and Data Fusion".

The dataset is available for download in:
M. Girolami, F. Furfari, P. Barsocchi, and F. Mavilia. (2023). A Bluetooth 5.1 Dataset Based on Angle of Arrival and RSS for Indoor Localization. Zenodo. https://doi.org/10.5281/zenodo.7759557

The 'calibration' and 'mobility' folders from Girolami et al. must be copied into the 'Dataset' folder in this repository. Ensure that the file paths are correctly specified in config.yaml

This code was tested by creating a virtual environment with Python 3.12.7. The requirements.txt file lists all the packages needed to generate the IPS algorithms.