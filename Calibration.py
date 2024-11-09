import yaml
import pandas as pd
import dataProcessing as dP
import matplotlib.pyplot as plt

def RSSI_model():
    print('Running Calibration Program ...')
    
    
    # Open Configrations File'
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    
    # Read dataset and transform to Pandas Dataframe
    beacons_column_names = ["TimeStamp", "TagID", "1stP","AoA_az", "AoA_el", "2ndP", "Channel", "AnchorID"] # Columns ID for Beacons Dataset
    gt_column_names = ["StartTime", "EndTime", "Xcoord","Ycoord"] # Columns ID for Ground Truth Dataset
    beacons_calibration = pd.read_csv(config['file_paths']['calibration']['beacons'], header=None, names=beacons_column_names) # DF of Beacons Calibration
    gt_calibration = pd.read_csv(config['file_paths']['calibration']['gt'], header=None, names=gt_column_names) # DF of GT Calibration
    
    # Iterate over each row in gt_calibration and apply the coordinates within the specified timestamp range
    for _, row in gt_calibration.iterrows():
        # Use a mask to find matching timestamps in `beacons_calibration`
        mask = (beacons_calibration['TimeStamp'] >= row['StartTime']) & (beacons_calibration['TimeStamp'] <= row['EndTime'])
        beacons_calibration.loc[mask, ['Xcoord', 'Ycoord']] = row[['Xcoord', 'Ycoord']].values
        
    # Remove rows in `beacons_calibration` where coordinates were not assigned
    beacons_calibration.dropna(subset=['Xcoord', 'Ycoord'], inplace=True)
    
    # Group the beacons_calibration dataframe by AnchorID
    beacons_calibration_by_anchor = {anchor_id: df for anchor_id, df in beacons_calibration.groupby("AnchorID")}
    
    # Initialize a dictionary to store the DataFrames
    beacons_calibration_dict = {}
    df_mean_dict = {}
    pathloss_dict = {}
    
    print('PATHLOSS COEFFICIENT VALUES:')
    # Access data for anchors 6501, 6502, 6503, 6504
    for anchor_id in ['a6501', 'a6502', 'a6503', 'a6504']:
        # Get coordinates and reference coordinates for the current anchor
        coordinates, reference_coordinates, not_used = dP.get_anchor_data(anchor_id, config)

        # Height difference between anchor and tag
        dH = coordinates[2] - reference_coordinates[2]

        # Get the corresponding beacons calibration dataframe for the current anchor
        beacons_df = beacons_calibration_by_anchor.get(int(anchor_id[1:]))
        
        # If the anchor exists in the dataframe, calculate the distance
        if beacons_df is not None:
            
            #calculate real distance
            beacons_df = dP.calculate_real_distance_df(beacons_df, anchor_id, coordinates, reference_coordinates, dH)
            
            #calcualte the mean measuremnts for each position
            mean_df = dP.mean_calibration(beacons_df, config)
            
            #calculate the pathloss coefficient
            n = dP.pathloss_calculation(mean_df, config['additional']['polarization'], reference_coordinates, coordinates, dH)
            
            #Calculate the RSSI model
            dP.rssi_model(mean_df, config['additional']['polarization'], coordinates, n, reference_coordinates, dH)
            
            # Update dataframes
            beacons_calibration_dict[anchor_id] = beacons_df
            
            df_mean_dict[anchor_id] = mean_df
            
            pathloss_dict[anchor_id] = n
            
            print(f'Anchor: {anchor_id[1:]} --- PLc: {n}')
            
            
    # Plot graphics: use the config.yaml to determine whether to display no plots, only the first plot, or all RSSI models.
    
    # Define the aspect ratio
    aspect_ratio = 16 / 9
    # Set the figure size based on the aspect ratio
    fig_width = 8  # You can choose any width
    fig_height = fig_width / aspect_ratio
    
    #for anchor_id in ['a6501', 'a6502', 'a6503', 'a6504']:
        
    if config['additional']['plot_all_RSSI'] == False and config['additional']['plot_first_RSSI'] == False:
        print('RSSI plot not selected in config.yaml')
        
    else:
        if config['additional']['plot_all_RSSI'] == True:
            print('Plotting RSSI for all anchors')
            plot_anchors = ['a6501', 'a6502', 'a6503', 'a6504']
        else:
            print('Plotting RSSI for Anchor 6501 only')
            plot_anchors = ['a6501']
            
        for anchor_id in plot_anchors:    
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                        
            x_plot = beacons_calibration_dict[anchor_id].iloc[::60, 10].values / 100  # Slicing every 60th value and dividing by 100 to transform in meters
            y_plot = beacons_calibration_dict[anchor_id].iloc[::60, 5].values  # Slicing every 60th value for RSSI
            plt.plot(x_plot, y_plot,'o')
            
            plt.plot(df_mean_dict[anchor_id]["D_real"]/100, df_mean_dict[anchor_id][config['additional']['polarization']],'r*') # distance (meters) x mean RSSI
            
            df_mean_dict[anchor_id].sort_values(by=['D_real'], inplace=True)
            plt.plot(df_mean_dict[anchor_id]["D_real"]/100, df_mean_dict[anchor_id]["RSSImodel"],'k', linewidth=3) # # distance (meters) x RSSI model
            
            plt.legend(['RSSI Measurements','Mean RSSI', 'RSSI Model'],loc=1, fontsize=11)
            plt.xlabel("Distance [m]",fontsize=12)
            plt.ylabel("RSSI [dBm]", fontsize=12)
            plt.title(f"RSSI x Distance - Anchor {anchor_id[1:]}", fontsize=14)
            plt.grid()
            plt.show()