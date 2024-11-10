import yaml
import pandas as pd
import numpy as np
import dataProcessing as dP
import matplotlib.pyplot as plt
import dataProcessing as dP
import findPositions as fP
from matplotlib.lines import Line2D  # Import Line2D for custom legend

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
        
        #get the initial movement positioning
        if case == 1:
            initial_position = config['kalman_filter']['case1_initial_pos']
            plot_start_label = [1.90,5.00]
            plot_stop_label = [9.12,0.90]
        elif case == 2:
            initial_position = config['kalman_filter']['case2_initial_pos']
            plot_start_label = [9.10,0.80]
            plot_stop_label = [1.90,0.80]
        else:
            initial_position = config['kalman_filter']['case3_initial_pos']
            plot_start_label = [1.90,5.00]
            plot_stop_label = [9.12,0.90]
        
        for run in range(1, 5):
            beacons_data, gt_data = dP.load_data(case, run, config, beacons_column_names, gt_column_names) #load dataset
            beacons_data, gt_data = dP.get_trajectory(beacons_data, gt_data, config) #interpolate points between GT
            
            beacons_data_runs[f'run{run}'] = beacons_data
            gt_data_runs[f'run{run}'] = gt_data

    else:
        print("Invalid Case")
        exit()


    # Vector to save results of each method
    all_errors_MLT = []
    mean_errors_MLT = []
    all_errors_Trigonometry = []
    mean_errors_Trigonometry = []
    all_errors_Triangulation = []
    mean_errors_Triangulation  = []
    all_errors_MLT_KF = []
    mean_errors_MLT_KF = []
    all_errors_Trigonometry_KF = []
    mean_errors_Trigonometry_KF = []
    all_errors_Triangulation_KF = []
    mean_errors_Triangulation_KF = []
    all_errors_ARFL = []
    mean_errors_ARFL = []
    
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
        
        #############################################################################################################3
        
        '''Get estimate position by AoA+RSSI (Trigonometry)'''
        df_posTrigonometry = fP.trigonometry(mean_data, config, df_posMLT)
        
        #Calculate the error
        mean_error_posTrigonometry, df_all_error_posTrigonometry= dP.distance_error(df_posTrigonometry)
        
        # Append the DataFrame of errors to the list - used latter to get the CDF
        all_errors_Trigonometry.append(df_all_error_posTrigonometry)
        
        # Append the average errors to the list
        mean_errors_Trigonometry.append(mean_error_posTrigonometry)
        
        ################################################################################################################
        
        '''Get estimate position by AoA-only (Triangulation)'''
        
        df_posTriangulation = fP.triangulation(mean_data, config)
        
        #Convert to float
        df_posTriangulation['Xest'] = df_posTriangulation['Xest'].astype(float) 
        df_posTriangulation['Yest'] = df_posTriangulation['Yest'].astype(float)
        
        #Calculate the error
        mean_error_posTriangulation, df_all_error_posTriangulation= dP.distance_error(df_posTriangulation)
        
        # Append the DataFrame of errors to the list - used latter to get the CDF
        all_errors_Triangulation.append(df_all_error_posTriangulation)
        
        # Append the average errors to the list
        mean_errors_Triangulation.append(mean_error_posTriangulation)
        
        ##################################################################################################################3
        '''Estime positions using the results from previously methods with Kalman Filter'''
        # Create measurements matrices using the helper function
        zk_posMLT = dP.create_measurement_matrix(df_posMLT)
        zk_posTrigonometry = dP.create_measurement_matrix(df_posTrigonometry)
        zk_posTriangulation = dP.create_measurement_matrix(df_posTriangulation)
        
        # Positioning obtained with multilateration + Kalman filter
        xk_MLT = fP.kalman_filter(zk_posMLT, config, initial_position, config['kalman_filter']['R_MLT'])
        df_posMLT_KF = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        df_posMLT_KF['Xreal'] = df_posMLT['Xreal']
        df_posMLT_KF['Yreal'] = df_posMLT['Yreal']
        df_posMLT_KF['Xest'] = xk_MLT[:,0]
        df_posMLT_KF['Yest'] = xk_MLT[:,2]
        
        #Calculate the error
        mean_error_posMLT_KF, df_all_error_posMLT_KF = dP.distance_error(df_posMLT_KF)
        
        # Append the DataFrame of errors to the list - used latter to get the CDF
        all_errors_MLT_KF.append(df_all_error_posMLT_KF)
        
        # Append the average errors to the list
        mean_errors_MLT_KF.append(mean_error_posMLT_KF)
        
        ########################################################################################################################3
         # Positioning obtained with Trigonometry + Kalman filter
        xk_Trigonometry = fP.kalman_filter(zk_posTrigonometry, config, initial_position, config['kalman_filter']['R_AoA_RSSI'])
        df_posTrigonometry_KF = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        df_posTrigonometry_KF['Xreal'] = df_posTrigonometry['Xreal']
        df_posTrigonometry_KF['Yreal'] = df_posTrigonometry['Yreal']
        df_posTrigonometry_KF['Xest'] = xk_Trigonometry[:,0]
        df_posTrigonometry_KF['Yest'] = xk_Trigonometry[:,2]
        
        #Calculate the error
        mean_error_posTrigonometry_KF, df_all_error_posTrigonometry_KF = dP.distance_error(df_posTrigonometry_KF)
        
        # Append the DataFrame of errors to the list - used latter to get the CDF
        all_errors_Trigonometry_KF.append(df_all_error_posTrigonometry_KF)
        
        # Append the average errors to the list
        mean_errors_Trigonometry_KF.append(mean_error_posTrigonometry_KF)
        
        #############################################################################################################################
        # Positioning obtained with Triangulation + Kalman filter
        xk_Triangulation = fP.kalman_filter(zk_posTriangulation, config, initial_position, config['kalman_filter']['R_AoA_only'])
        df_posTriangulation_KF = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        df_posTriangulation_KF['Xreal'] = df_posTriangulation['Xreal']
        df_posTriangulation_KF['Yreal'] = df_posTriangulation['Yreal']
        df_posTriangulation_KF['Xest'] = xk_Triangulation[:,0]
        df_posTriangulation_KF['Yest'] = xk_Triangulation[:,2]
        
        #Calculate the error
        mean_error_posTriangulation_KF, df_all_error_posTriangulation_KF = dP.distance_error(df_posTriangulation_KF)
        
        # Append the DataFrame of errors to the list - used latter to get the CDF
        all_errors_Triangulation_KF.append(df_all_error_posTriangulation_KF)
        
        # Append the average errors to the list
        mean_errors_Triangulation_KF.append(mean_error_posTriangulation_KF)
        
        ########################################################################################################################
        '''Estimate position using ARFL fusion with AoA+RSSI (Trigonometry) and AoA-only (Triangulation)'''
        xk_ARFL = fP.ARFL_fusion(zk_posTrigonometry, zk_posTriangulation, config['kalman_filter']['R_AoA_RSSI'] , config['kalman_filter']['R_AoA_only'], initial_position, config)
        
        df_posARFL = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        df_posARFL['Xreal'] = df_posTriangulation['Xreal']
        df_posARFL['Yreal'] = df_posTriangulation['Yreal']
        df_posARFL['Xest'] = xk_ARFL[:,0]
        df_posARFL['Yest'] = xk_ARFL[:,2]
        
        #Calculate the error
        mean_error_posARFL, df_all_error_posARFL = dP.distance_error(df_posARFL)
        
        # Append the DataFrame of errors to the list - used latter to get the CDF
        all_errors_ARFL.append(df_all_error_posARFL)
        
        # Append the average errors to the list
        mean_errors_ARFL.append(mean_error_posARFL)
        
        
        #######################################################################################################
        '''Plot real and estimate positions'''
        # Estimate path plot - Only plot if True in config.yaml
        if config['additional']['plot_first_path'] == False and config['additional']['plot_all_paths'] == False:
            plot_path = False
        else:
            if config['additional']['plot_all_paths'] == True:
                plot_path = True
            else:
                if run == 1:
                    plot_path = True
                else:
                    plot_path = False
        
        if plot_path:
            print(f'Plotting path for run {run}')
            plt.figure()
            plt.ylim(-2.00, 8.00)
            plt.xlim(-2.00,14.00)
            plt.plot([0,12.00,12.00,0,0],[0,0,6.00,6.00,0],'k')
            # Show the major grid and style it slightly.
            plt.grid(which='major', color='#DDDDDD', linewidth=1)
            # Show the minor grid as well. Style it in very light gray as a thin,
            # dotted line.
            plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.8)
            # Make the minor ticks and gridlines show.
            plt.minorticks_on()
            #plot anchors
            plt.plot([0,6.00,12.00,6.00],[3.00,0,3.00,6.00],'ro', markersize=8)
            plt.text(-1.30,2.85,'6501')
            plt.text(5.50,-0.50,'6502')
            plt.text(12.25,2.85,'6503')
            plt.text(5.50,6.20,'6504')
            plt.text(plot_start_label[0],plot_start_label[1],'Start')
            plt.text(plot_stop_label[0],plot_stop_label[1],'Stop')
            plt.xlabel('x [m]', loc='right', fontsize = 12)
            plt.ylabel('y [m]', loc='top', fontsize = 12)
            
            for i in range (0,len(df_posARFL),3):
               plt.plot(df_posARFL.iloc[i,0]/100,df_posARFL.iloc[i,1]/100, 'b*') # real trajectory
               plt.plot(df_posTriangulation.iloc[i,2]/100,df_posTriangulation.iloc[i,3]/100, 'c.') #position by triangulation
               plt.plot(df_posTriangulation_KF.iloc[i,2]/100,df_posTriangulation_KF.iloc[i,3]/100, 'm.') #position by triangulation+KF
               plt.plot(df_posTrigonometry.iloc[i,2]/100,df_posTrigonometry.iloc[i,3]/100, 'y.') #position by trigonometry
               plt.plot(df_posTrigonometry_KF.iloc[i,2]/100,df_posTrigonometry_KF.iloc[i,3]/100, 'k.') #position by trigonometry+KF
               plt.plot(df_posARFL.iloc[i,2]/100,df_posARFL.iloc[i,3]/100, 'g.') #position by ARFL
               
               # distance errors lines
               plt.plot([df_posARFL.iloc[i,0]/100,df_posTriangulation_KF.iloc[i,2]/100], [df_posARFL.iloc[i,1]/100,df_posTriangulation_KF.iloc[i,3]/100],'m', linewidth=0.2) # lines real to triangulation+KF
               plt.plot([df_posARFL.iloc[i,0]/100,df_posTriangulation.iloc[i,2]/100], [df_posARFL.iloc[i,1]/100,df_posTriangulation.iloc[i,3]/100],'c', linewidth=0.2) # lines real to triangulation
               plt.plot([df_posARFL.iloc[i,0]/100,df_posTrigonometry_KF.iloc[i,2]/100], [df_posARFL.iloc[i,1]/100,df_posTrigonometry_KF.iloc[i,3]/100],'k', linewidth=0.2) # lines real to trigonometry
               plt.plot([df_posARFL.iloc[i,0]/100,df_posTrigonometry.iloc[i,2]/100], [df_posARFL.iloc[i,1]/100,df_posTrigonometry.iloc[i,3]/100],'y', linewidth=0.2) # lines real to trigonometry+KF
               plt.plot([df_posARFL.iloc[i,0]/100,df_posARFL.iloc[i,2]/100], [df_posARFL.iloc[i,1]/100,df_posARFL.iloc[i,3]/100],'g', linewidth=0.2) # lines real to ARFL
            
            plt.title(f'Case: {case} - Run: {run} -- Position Estimation')
            plt.legend(['Area','Anchors','Real Trajectory' , 'AoA-only', 'AoA-only+KF', 'AoA+RSSI','AoA+RSSI+KF', 'ARFL'],loc=1, fontsize='small')
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)
            #plt.legend(['Area','Âncoras','Trajetória Real'],loc=1, fontsize='small')
            plt.show(block=True)
            
        
    ###############################################################################################################################
    
    # Calculate the overall mean average error
    overall_mean_error_MLT = round(sum(mean_errors_MLT) / (len(mean_errors_MLT)*100), 2) #transform to meters and round
    print(f"\nMultilateration Average Error across all runs: {overall_mean_error_MLT}")
    overall_mean_error_Trigonometry = round(sum(mean_errors_Trigonometry) / (len(mean_errors_Trigonometry)*100), 2) #transform to meters and round
    print(f"AoA+RSSI Average Error across all runs: {overall_mean_error_Trigonometry}")
    overall_mean_error_Triangulation = round(sum(mean_errors_Triangulation) / (len(mean_errors_Triangulation)*100), 2) #transform to meters and round
    print(f"AoA-only Average Error across all runs: {overall_mean_error_Triangulation}\n")
    overall_mean_error_MLT_KF = round(sum(mean_errors_MLT_KF) / (len(mean_errors_MLT_KF)*100), 2) #transform to meters and round
    print(f"Multilateration+KF Average Error across all runs: {overall_mean_error_MLT_KF}")
    overall_mean_error_Trigonometry_KF = round(sum(mean_errors_Trigonometry_KF) / (len(mean_errors_Trigonometry_KF)*100), 2) #transform to meters and round
    print(f"AoA+RSSI+KF Average Error across all runs: {overall_mean_error_Trigonometry_KF}")
    overall_mean_error_Triangulation_KF = round(sum(mean_errors_Triangulation_KF) / (len(mean_errors_Triangulation_KF)*100), 2) #transform to meters and round
    print(f"AoA-only Average Error across all runs: {overall_mean_error_Triangulation_KF}\n")
    overall_mean_error_ARFL = round(sum(mean_errors_ARFL) / (len(mean_errors_ARFL)*100), 2) #transform to meters and round
    print(f"ARFL Average Error across all runs: {overall_mean_error_ARFL}\n")
    
    
    # Concatenate all DataFrames of errors into a single DataFrame
    df_error_MLT_all = pd.concat(all_errors_MLT, ignore_index=True)
    df_error_Trigonometry_all = pd.concat(all_errors_Trigonometry, ignore_index=True)
    df_error_Triangulation_all = pd.concat(all_errors_Triangulation, ignore_index=True)
    df_error_MLT_KF_all = pd.concat(all_errors_MLT, ignore_index=True)
    df_error_Trigonometry_KF_all = pd.concat(all_errors_Trigonometry_KF, ignore_index=True)
    df_error_Triangulation_KF_all = pd.concat(all_errors_Triangulation_KF, ignore_index=True)
    df_error_ARFL_all = pd.concat(all_errors_ARFL, ignore_index=True)
    
    
    ##############################################################################
    '''Start of plots (Erro Bargraph and CDF)'''
    
     # Define the aspect ratio
    aspect_ratio = 16 / 9
    # Set the figure size based on the aspect ratio
    fig_width = 8  # You can choose any width
    fig_height = fig_width / aspect_ratio
    
    if config['additional']['plot_error_bargraph'] == True:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        methods = ['MLT', 'MLT+KF', 'AoA+RSSI', 'AoA+RSSI+KF', 'AoA', 'AoA+KF', 'ARFL']
        results = [overall_mean_error_MLT, overall_mean_error_MLT_KF, overall_mean_error_Trigonometry, overall_mean_error_Trigonometry_KF, overall_mean_error_Triangulation, overall_mean_error_Triangulation_KF, overall_mean_error_ARFL]
        bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'tab:green' ]
        bar_labels = ['Without Filter', 'With KF', '_Without Filter', '_White KF' ,'_Without Filter', '_KF', 'ARFL']
        plt.xticks(range(len(methods)),('MLT', 'MLT+KF', 'AoA+RSSI', 'AoA+RSSI+KF', 'AoA', 'AoA+KF', 'ARFL'), rotation=30,fontsize=11)
        ax.bar(methods, results, label=bar_labels, color=bar_colors)
        for i in range (len(methods)):
            plt.text(i, results[i]/2, (results[i]), ha = 'center', va='center')

        # plt.plot(metodos, resultados, 'k', linewidth=2.0)
        ax.set_ylabel('Distance Error [m]', fontsize=12)
        ax.set_title(f'Case: {case} -- Distance Error x Method', fontsize=14)
        ax.legend(title='Method', title_fontsize=12)
        plt.tick_params(axis='y', labelsize=12)
        plt.show()
            
    if config['additional']['plot_CDF'] == True:
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Config Y-limit and percentiles
        ax.set_ylim(0, 1)
        percentile_values = [0.05, 0.25, 0.5, 0.75, 0.95]
        ax.set_yticks(percentile_values)

        # Function do generate percentiles
        def plot_cumulative_hist(data, color, label):
            counts, bin_edges = np.histogram(data/100, bins=100, density=True)
            cdf = np.cumsum(counts) / np.sum(counts)  # Calculate cumulative distribution
            plt.step(bin_edges[:-1], cdf, where='post', color=color, label=label, linewidth=2)  # Skip the last bin edge

            # Calculate and annotate percentiles
            percentiles = [5, 25, 50, 75, 95]
            for p in percentile_values:
                # Get the corresponding x value (distance error)
                value = np.percentile(data, p) / 100  # Divide by 100 to match scaling in plot
                
                # Find the corresponding y value on the CDF (percentile in terms of probability)
                y_value = p / 100.0
                
                # Plot the horizontal line from the y-axis (at y_value) to the CDF curve at 'value'
                plt.hlines(y=y_value, xmin=0, xmax=value, color='grey', linestyle='--', linewidth=1)
                              
        # Plot the first cumulative histogram
        plot_cumulative_hist(df_error_ARFL_all, color='green', label='alg1')

        # Plot the second cumulative histogram
        plot_cumulative_hist(df_error_Triangulation_KF_all, color='purple', label='alg2')

        # Plot the third cumulative histogram
        plot_cumulative_hist(df_error_Trigonometry_KF_all, color='black', label='alg3')

        # Add labels and title
        plt.xlabel('Distance Error [m]', fontsize=14)
        plt.ylabel('Cumulative Probability', fontsize=14)
        plt.title(f'Case: {case} -- Cumulative Distribution Error', fontsize=14)
        plt.xlim(left=0, right=3)  # Ajuste limite do eixo x de 0 a 3
        plt.ylim(0, 1)  # Limite do eixo y de 0 a 1

        plt.grid(True, axis='y')  # Grade apenas no eixo y

        # Create custom legend lines (Line2D objects)
        custom_lines = [Line2D([0], [0], color='green', lw=2),
                        Line2D([0], [0], color='purple', lw=2),
                        Line2D([0], [0], color='black', lw=2)]

        # Add a legend using the custom lines
        plt.legend(custom_lines, ['ARFL', 'AoA+KF', 'AoA+RSSI+KF'], handlelength=2, loc=4, fontsize=14)
        plt.tick_params(axis='y', labelsize=12)
        plt.tick_params(axis='x', labelsize=12)

        # Show the plot
        plt.show()
        print()
    
        