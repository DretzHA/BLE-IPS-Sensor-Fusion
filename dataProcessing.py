import numpy as np
import math
import pandas as pd
import sympy as sym

# Helper function to load datasets for movement scenarios
def load_data(case, run, config, beacons_column_names, gt_column_names):
    """Load the beacons and ground truth data for a specific case and run."""
    beacons_file_path = config['file_paths']['mobility'][f'use_case_{case}'][f'run{run}']['beacons']
    gt_file_path = config['file_paths']['mobility'][f'use_case_{case}'][f'run{run}']['gt']
    
    beacons_data = pd.read_csv(beacons_file_path, header=None, names=beacons_column_names)
    gt_data = pd.read_csv(gt_file_path, header=None, names=gt_column_names)
    
    return beacons_data, gt_data

# Function to get coordinates, PLc and reference coordinates for a given anchor ID
def get_anchor_data(anchor_id, config):
    anchor = next(anchor for anchor in config['anchors'] if anchor['id'] == anchor_id)
    return anchor['coordinates'], anchor['ref_coordinates'], anchor['alpha']

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
    
# Function to  interpolate the trajectory of a moving object by processing ground truth and beacon data.
def get_trajectory(beacons_movement, gt_movement, config):
    
    # Initialize position and time variables
    initial_x, initial_y = gt_movement.iloc[0, 2], gt_movement.iloc[0, 3]
    initial_time = gt_movement.iloc[0, 0]
    final_time = gt_movement.iloc[-1, 0]
    gt_movement['Vx'] = 0.0
    gt_movement['Vy'] = 0.0
    
    # Calculate velocities for gt_movement
    for i in range(1, len(gt_movement)):
        
        prev_end_time = gt_movement.iloc[i - 1, 1]
        time_diff = (gt_movement.iloc[i, 0] - (prev_end_time if not pd.isna(prev_end_time) else gt_movement.iloc[i - 1, 0])) / 1000
        gt_movement.iloc[i, 4] = (gt_movement.iloc[i, 2] - gt_movement.iloc[i - 1, 2]) / 100 / time_diff  # Vx in m/s
        gt_movement.iloc[i, 5] = (gt_movement.iloc[i, 3] - gt_movement.iloc[i - 1, 3]) / 100 / time_diff  # Vy in m/s
        
    # Filter beacons_movement by initial and final time
    beacons_movement = beacons_movement[(beacons_movement['TimeStamp'] >= initial_time) & 
                                        (beacons_movement['TimeStamp'] <= final_time)].copy()
    
    # Discretize time for beacons_movement
    Tdisc = config['kalman_filter']['delta_T']  # Discretization time in seconds
    beacons_movement['InitialTime'] = ((beacons_movement['TimeStamp'] - initial_time) / 1000 / Tdisc).astype(int)

    # Calculate real position over time
    Xr, Yr = [], []
    actual_x, actual_y = initial_x, initial_y
    old_timer = initial_time
    
    for actual_time in beacons_movement['TimeStamp']:
        for k in range(1, len(gt_movement)):
            gt_start_time, gt_end_time = gt_movement.iloc[k, 0], gt_movement.iloc[k - 1, 1]
            if actual_time <= gt_start_time:
                actual_vx = gt_movement.iloc[k, 4] if pd.isna(gt_end_time) or actual_time > gt_end_time else 0
                actual_vy = gt_movement.iloc[k, 5] if pd.isna(gt_end_time) or actual_time > gt_end_time else 0
                break

        # Update position based on velocity and elapsed time
        time_elapsed = (actual_time - old_timer) / 1000
        actual_x += actual_vx * time_elapsed * 100  # Convert to cm
        actual_y += actual_vy * time_elapsed * 100  # Convert to cm
        old_timer = actual_time

        Xr.append(int(actual_x))
        Yr.append(int(actual_y))

    beacons_movement['Xcoord'] = Xr
    beacons_movement['Ycoord'] = Yr

    return beacons_movement, gt_movement

# Function to get the mean measurements for each mobility points
def mean_mobility(dataframe, config):
    # Calculate RSSILin for the entire DataFrame
    dataframe['RSSILin'] = np.power(10, (dataframe['2ndP'] - 30) / 10)

    # Group by 'InitialTime' and calculate mean values for each group
    grouped = dataframe.groupby('InitialTime').agg(
        Xreal=('Xcoord', 'mean'),
        Yreal=('Ycoord', 'mean'),
        Dreal=('Distance', 'mean'),
        Azim=('AoA_az', 'mean'),
        Elev=('AoA_el', 'mean'),
        MeanRSSI=('RSSILin', 'mean')  # Using linear RSSI mean initially
    ).reset_index()

    # Convert MeanRSSI back to dBm using log scale and add offset
    grouped['MeanRSSI'] = 10 * np.log10(grouped['MeanRSSI']) + 30

    # Adjust InitialTime to represent the time in seconds
    grouped['InitialTime'] = grouped['InitialTime'] * config['kalman_filter']['delta_T']  # Discretization time in seconds

    return grouped[['InitialTime', 'Xreal', 'Yreal', 'Dreal', 'Azim', 'Elev', 'MeanRSSI']]

#Function to correct the size of the dataframes 
def df_correct_sizes(df6501, df6502, df6503, df6504):
    # List of DataFrames
    dfs = [df6501, df6502, df6503, df6504]
    
    # Find the maximum number of rows across DataFrames
    total_row = max(len(df) for df in dfs)
    
    # Ensure each DataFrame has the same number of rows
    for i, df in enumerate(dfs):
        if len(df) < total_row:
            # Calculate the number of rows to add
            rows_to_add = total_row - len(df)
            # Concatenate empty rows to match `total_row` length
            dfs[i] = pd.concat([df, pd.DataFrame(np.nan, index=range(rows_to_add), columns=df.columns)], ignore_index=True)
        
        # Interpolate missing data
        dfs[i].interpolate(method='linear', inplace=True)
    
    return dfs[0], dfs[1], dfs[2], dfs[3]

# Function to correct the circles radius
def adjust_circle_eccentric(radius1, radius2, distance_between_anchors):
    
    if radius1>radius2: 
        while int(radius1)>int((distance_between_anchors+radius2)): # While do not intersect
            radius1 = radius1-(radius1/(radius2+radius1))*(radius1-distance_between_anchors-radius2) # reduce radius 1
            radius2 = radius2+(radius2/(radius2+radius1))*(radius1-distance_between_anchors-radius2) # grow radius 2
            radius1 = radius1*0.9 #Scalling Factor
            radius2 = radius2*1.1 #Scalling Factor
            
    elif radius1<radius2:
        while int(radius2)>int((distance_between_anchors+radius1)): # While do not intersect
            radius2 = radius2-(radius2/(radius2+radius1))*(radius2-distance_between_anchors-radius1) # reduce radius 2
            radius1 = radius1+(radius1/(radius2+radius1))*(radius2-distance_between_anchors-radius1) # grow radius 1
            radius1 = radius1*1.1 #Scalling Factor
            radius2 = radius2*0.9 #Scalling Factor

    return radius1, radius2

# Function to correct the circles radius
def adjust_separate_circle_radii(d1, d2, d12):
    
    while int(d12)>int((d1+d2)): # While do not intersect
        d1 = (d1/(d1+d2))*d12  # grow radius 1
        d2 = (d2/(d1+d2))*d12 # grow radius 2
        d1 = d1*1.1  #Scalling Factor
        d2 = d2*1.1 #Scalling Factor

    return d1, d2

# Function to calculate the error
def distance_error(dataframe):
    
    # Calculate Euclidean distance error for each row (vectorized operation)
    errors = np.sqrt((dataframe.iloc[:, 0] - dataframe.iloc[:, 2])**2 + (dataframe.iloc[:, 1] - dataframe.iloc[:, 3])**2)
    
    # Create DataFrame for individual errors
    df_error = pd.DataFrame({'Erro': errors})

    # Calculate average error
    mean_error = errors.mean()

    return mean_error, df_error

# Helper function to create the measurement matrix
def create_measurement_matrix(position_data, x_key='Xest', y_key='Yest'):
    x_values = position_data[x_key]
    y_values = position_data[y_key]
    return np.array([x_values, y_values]).transpose()