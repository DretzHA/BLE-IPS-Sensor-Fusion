import numpy as np
import math
import pandas as pd
import sympy as sym

# Function to get coordinates and reference coordinates for a given anchor ID
def get_anchor_coordinates_data(anchor_id, config):
    anchor = next(anchor for anchor in config['anchors'] if anchor['id'] == anchor_id)
    return anchor['coordinates'], anchor['ref_coordinates']

# Function to calculate the real distance between calibration points and anchor
def calculate_real_distance_df(df, anchor_id, coordinates, reference_coordinates, dH):
    df["Distance"] = np.sqrt(
        (df["Xcoord"] - coordinates[0])**2 +
        (df["Ycoord"] - coordinates[1])**2 +
        dH**2
    )
    return df

#Function to get the mean measurements for each calibration points
def mean_calibration(dataframe, config):
    beginX = config['calibration']['begin_X']
    endX = config['calibration']['end_X']
    beginY = config['calibration']['begin_Y']
    endY = config['calibration']['end_Y']
    zscore = config['calibration']['Z_score']
    Plzt = config['additional']['polarization']
    
    # Calculate RSSI Linear in a vectorized manner
    dataframe['RSSILin'] = np.power(10, (dataframe[Plzt] - 30) / 10)
    
    # Create Z-Score column in one go (avoiding groupby within the loop)
    dataframe['Z-Score'] = dataframe.groupby(['Xcoord', 'Ycoord'])[Plzt].transform(lambda x: (x - x.mean()) / x.std()).abs()
    
    # Filter based on Z-Score once (avoid filtering in the loop)
    dataframe = dataframe[dataframe['Z-Score'] <= zscore]
    
    # Prepare the results list instead of appending rows one by one
    result_rows = []
    
    # Iterate over the grid defined by (beginX, endX, beginY, endY)
    for i_x in range(beginX, endX + 1, 60):
        for i_y in range(beginY, endY + 1, 60):
            # Precompute the subset of the dataframe for this specific (i_x, i_y)
            subset = dataframe[(dataframe['Xcoord'] == i_x) & (dataframe['Ycoord'] == i_y)]
            
            # Calculate the means of the azimuth, elevation, and RSSI linear values
            mean_az = subset['AoA_az'].mean()
            mean_el = subset['AoA_el'].mean()
            mean_RSSI = subset['RSSILin'].mean()
            
            # Convert the mean RSSI back to dB
            mean_RSSI = 10 * math.log10(mean_RSSI) + 30
            
            # Add a new row to the result list
            result_rows.append([i_x, i_y, mean_az, mean_el, mean_RSSI])
    
    # Convert the result list to a DataFrame at the end
    mean_df = pd.DataFrame(result_rows, columns=['Xreal', 'Yreal', 'Azim', 'Elev', Plzt])
    
    return mean_df

#Function to calculate the pathloss coefficient for each anchor
def pathloss_calculation(dataframe, Plzt, reference_coordinates, coordinates, dH):
    
    d0_x = reference_coordinates[0]
    d0_y = reference_coordinates[1]
    X_a = coordinates[0]
    Y_a = coordinates[1]
    
    # Calculate the reference RSSI at position (d0_x, d0_y)
    rssi_d0 = dataframe.loc[(dataframe['Xreal'] == d0_x) & (dataframe['Yreal'] == d0_y), Plzt].mean()
    
    # Calculate the reference distance (d0) from anchor position
    d0 = np.sqrt((d0_x - X_a)**2 + (d0_y - Y_a)**2 + dH**2)
    
    # Calculate the distances to anchor once and store them
    distances = np.sqrt((dataframe['Xreal'] - X_a)**2 + (dataframe['Yreal'] - Y_a)**2 + dH**2)
    
    # Use vectorized operations to calculate RSSI model
    # Define the path loss model for each point based on the distance
    model_rssi = rssi_d0 - 10 * sym.Symbol('n') * np.log10(distances / d0)
    
    # Compute the path loss exponent (n) using least squares
    # Minimize the sum of squared errors between the model and actual RSSI values
    def objective_function(n_value):
        # Calculate the error between model and actual RSSI
        model = rssi_d0 - 10 * n_value * np.log10(distances / d0)
        error = np.sum((dataframe[Plzt] - model) ** 2)
        return error

    # Initial guess for the path loss exponent (n)
    initial_guess = 2.0
    
    # Minimize the error using scipy optimization (least squares method)
    from scipy.optimize import minimize
    result = minimize(objective_function, initial_guess)
    
    # Optimal path loss exponent (n)
    n = result.x[0]
    
    # Add the real distance column to the dataframe
    dataframe['D_real'] = distances
    
    return n

#Function to get the RSSI model utilizing the pathloss
def rssi_model(dataframe, Plzt, coordinates, n, reference_coordinates, dH):
    
    d0_x = reference_coordinates[0]
    d0_y = reference_coordinates[1]
    X_a = coordinates[0]
    Y_a = coordinates[1]
    
    
    # Calculate reference distance once
    d0 = np.sqrt((d0_x - X_a)**2 + (d0_y - Y_a)**2 + dH**2)
    
    # Calculate the reference RSSI once
    rssi_d0 = dataframe.query(f'Xreal == {d0_x} and Yreal == {d0_y}')[Plzt].mean()
    
    # Vectorized RSSI model calculation
    Dreal = dataframe['D_real']
    dataframe['RSSImodel'] = rssi_d0 - (10 * n) * np.log10(Dreal / d0)
    
    # Estimate distance by log-distance model
    dataframe["Dest_RSSI"] = d0 * 10**((rssi_d0 - dataframe[Plzt]) / (10 * n))
    