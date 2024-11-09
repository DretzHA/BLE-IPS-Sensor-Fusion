import yaml
import pandas as pd
import numpy as np
import dataProcessing as dP
import matplotlib.pyplot as plt
import dataProcessing as dP
import findPositions as fP

def run(case):
    
    # Open Configrations File'
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    # Read dataset and transform to Pandas Dataframe
    beacons_column_names = ["TimeStamp", "TagID", "1stP","AoA_az", "AoA_el", "2ndP", "Channel", "AnchorID"] # Columns ID for Beacons Dataset
    gt_column_names = ["StartTime", "EndTime", "Xcoord","Ycoord"] # Columns ID for Ground Truth Dataset
    
    # Check case validity
    if case in [1, 2, 3]:
        print(f"Case {case}")

        # Load data for runs 1 to 4 and store in dictionaries
        beacons_data_runs = {}
        gt_data_runs = {}
        
        for run in range(1, 5):
            beacons_data, gt_data = dP.load_data(case, run, config, beacons_column_names, gt_column_names) #load dataset
            beacons_data, gt_data = dP.get_trajectory(beacons_data, gt_data, config) #interpolate points between GT
            
            beacons_data_runs[f'run{run}'] = beacons_data
            gt_data_runs[f'run{run}'] = gt_data

    else:
        print("Invalid Case")
        exit()


    all_errors_MLT = []
    mean_errors_MLT = []
    all_errors_Trigonometry = []
    mean_errors_Trigonometry = []
    for run in range(1, 5):
        print(f'Executing RUN {run}')
        filtered_data = {}
        mean_data = {}
        for anchor_id in ['a6501', 'a6502', 'a6503', 'a6504']:
            # Get coordinates and reference coordinates for the current anchor
            coordinates, reference_coordinates, alpha = dP.get_anchor_data(anchor_id, config)
            
            # Height difference between anchor and tag
            dH = coordinates[2] - reference_coordinates[2]
        
            # Filter the data for each anchor ID and store in the dictionary
            filtered_data[f'{anchor_id}'] = beacons_data_runs[f'run{run}'][beacons_data_runs[f'run{run}']["AnchorID"] == int(anchor_id[1:])].copy()
        
            #calculate real distance - not used, just calculating to maintain the size of dataframe as the first version of the codes
            filtered_data[f'{anchor_id}'] = dP.calculate_real_distance_df(filtered_data[f'{anchor_id}'], anchor_id, coordinates, reference_coordinates, dH)   

            # Obtain the mean data for each position (discretized time)
            mean_data[f'{anchor_id}'] = dP.mean_mobility(filtered_data[f'{anchor_id}'], config)

            #Calculate the RSSI utilizing the distance to initial points using the PLc obtained by the calibration code    
            mean_data[f'{anchor_id}']['Dest_RSSI'] = mean_data[f'{anchor_id}'].iloc[0,3] * np.power(10, ((mean_data[f'{anchor_id}'].iloc[0, 6] - mean_data[f'{anchor_id}']["MeanRSSI"]) / (10 * alpha)))


        #Verify the lenght of dataframes and interpolate data if are missing measurements
        mean_data[f'a6501'], mean_data[f'a6502'], mean_data[f'a6503'], mean_data[f'a6504'] = dP.df_correct_sizes(mean_data[f'a6501'], mean_data[f'a6502'], mean_data[f'a6503'], mean_data[f'a6504'])
      
      
        '''Here starts the algorithms to estimate the position'''
        
        '''Get estimate position by Multilateration'''
        
        #Calculate positioning with Multilateration Funcion
        df_posMLT = fP.multilateration(config, mean_data)
        
        #Convert to float
        df_posMLT['Xest'] = df_posMLT['Xest'].astype(float) 
        df_posMLT['Yest'] = df_posMLT['Yest'].astype(float)
        
        #Calculate the error
        mean_error_posMLT, df_all_error_posMLT= dP.distance_error(df_posMLT)
        
        # Append the DataFrame of errors to the list - used latter to get the CDF
        all_errors_MLT.append(df_all_error_posMLT)
        
        # Append the average errors to the list
        mean_errors_MLT.append(mean_error_posMLT)
        
        
        '''Get estimate position by AoA+RSSI (Trigonometry)'''
        df_posTrigonometry = fP.trigonometry(mean_data, config, df_posMLT)
        
        #Calculate the error
        mean_error_posTrigonometry, df_all_error_posTrigonometry= dP.distance_error(df_posTrigonometry)
        
        # Append the DataFrame of errors to the list - used latter to get the CDF
        all_errors_Trigonometry.append(df_all_error_posTrigonometry)
        
        # Append the average errors to the list
        mean_errors_Trigonometry.append(mean_error_posTrigonometry)
        
        
    # Calculate the overall mean average error
    overall_mean_error_MLT = round(sum(mean_errors_MLT) / (len(mean_errors_MLT)*100), 2) #transform to meters and round
    print(f"Multilateration Average Error across all runs: {overall_mean_error_MLT}")
    overall_mean_error_Trigonometry = round(sum(mean_errors_Trigonometry) / (len(mean_errors_Trigonometry)*100), 2) #transform to meters and round
    print(f"Aoa+RSSI Average Error across all runs: {overall_mean_error_Trigonometry}\n")
    
    
    # Concatenate all DataFrames of errors into a single DataFrame
    df_error_MLT_all = pd.concat(all_errors_MLT, ignore_index=True)
    df_error_Trigonometry_all = pd.concat(all_errors_Trigonometry, ignore_index=True)
    
    

    print()
    
        