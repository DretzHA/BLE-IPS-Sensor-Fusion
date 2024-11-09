import numpy as np
import pandas as pd
import dataProcessing as dP
import math

#Function to obtain position by multilateratio
def multilateration(config, mean_data):
    
    anchors_coordinates = {}
    for anchor in config['anchors']: # getting the data for each anchors
        anchor_id = anchor['id']
        x, y, z = anchor['coordinates']  # get coordinates
        xr, xy, zr = anchor['ref_coordinates']  # get references coordinates
        anchors_coordinates[anchor_id] = {
            'x': x,
            'y': y,
            'z': z,
            'alpha': anchor['alpha'],
            'ref_coordinates': anchor['ref_coordinates']
        }   
        
    # Distance between anchors:
    d12 = np.sqrt((pow(anchors_coordinates['a6501']['x']-anchors_coordinates['a6502']['x'],2))+(pow(anchors_coordinates['a6501']['y']-anchors_coordinates['a6502']['y'],2)))
    d13 = np.sqrt((pow(anchors_coordinates['a6501']['x']-anchors_coordinates['a6503']['x'],2))+(pow(anchors_coordinates['a6501']['y']-anchors_coordinates['a6503']['y'],2)))
    d14 = np.sqrt((pow(anchors_coordinates['a6501']['x']-anchors_coordinates['a6504']['x'],2))+(pow(anchors_coordinates['a6501']['y']-anchors_coordinates['a6504']['y'],2)))
    d23 = np.sqrt((pow(anchors_coordinates['a6502']['x']-anchors_coordinates['a6503']['x'],2))+(pow(anchors_coordinates['a6502']['y']-anchors_coordinates['a6503']['y'],2)))
    d24 = np.sqrt((pow(anchors_coordinates['a6502']['x']-anchors_coordinates['a6504']['x'],2))+(pow(anchors_coordinates['a6502']['y']-anchors_coordinates['a6504']['y'],2)))
    d34 = np.sqrt((pow(anchors_coordinates['a6503']['x']-anchors_coordinates['a6504']['x'],2))+(pow(anchors_coordinates['a6503']['y']-anchors_coordinates['a6504']['y'],2)))

    df_MLT = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #Dataframe to save the results
    
    for i in range (len(mean_data['a6501'])): #loop through measurements
       
        #Distance with RSSI between anchors and tag
        d1 = mean_data['a6501'].iloc[i,7]
        d2 = mean_data['a6502'].iloc[i,7]
        d3 = mean_data['a6503'].iloc[i,7]
        d4 = mean_data['a6504'].iloc[i,7]
        
        
        '''Check if the radii of the circles formed by the distance and position of the anchor are eccentric
        or do not touch other circles. If so, make a correction by decreasing or increasing the distance.
        A loop is implemented to ensure that when a correction is made, other radii are not affected'''
        for k in range(0,100):
            
            d2,d3 = dP.adjust_circle_eccentric(d2,d3,d23)
            d2,d4 = dP.adjust_circle_eccentric(d2,d4,d24)
            d3,d4 = dP.adjust_circle_eccentric(d3,d4,d34)  
            d1,d2 = dP.adjust_circle_eccentric(d1,d2,d12)
            d1,d3 = dP.adjust_circle_eccentric(d1,d3,d13)
            d1,d4 = dP.adjust_circle_eccentric(d1,d4,d14)
                
            d2,d3 = dP.adjust_separate_circle_radii(d2,d3,d23)
            d2,d4 = dP.adjust_separate_circle_radii(d2,d4,d24)
            d3,d4 = dP.adjust_separate_circle_radii(d3,d4,d34)
            d1,d2 = dP.adjust_separate_circle_radii(d1,d2,d12)
            d1,d3 = dP.adjust_separate_circle_radii(d1,d3,d13)
            d1,d4 = dP.adjust_separate_circle_radii(d1,d4,d14)
            
            
        # Multilateration equations
        b1 = -pow(anchors_coordinates['a6501']['x'],2)-pow(anchors_coordinates['a6501']['y'],2)+pow(anchors_coordinates['a6504']['x'],2)+pow(anchors_coordinates['a6504']['y'],2)+pow(d1,2)-pow(d4,2)
        b2 = -pow(anchors_coordinates['a6502']['x'],2)-pow(anchors_coordinates['a6502']['y'],2)+pow(anchors_coordinates['a6504']['x'],2)+pow(anchors_coordinates['a6504']['y'],2)+pow(d2,2)-pow(d4,2)
        b3 = -pow(anchors_coordinates['a6503']['x'],2)-pow(anchors_coordinates['a6503']['y'],2)+pow(anchors_coordinates['a6504']['x'],2)+pow(anchors_coordinates['a6504']['y'],2)+pow(d3,2)-pow(d4,2)
       
        m11 = 2*(-anchors_coordinates['a6501']['x']+anchors_coordinates['a6504']['x'])
        m12 = 2*(-anchors_coordinates['a6501']['y']+anchors_coordinates['a6504']['y'])
        m21 = 2*(-anchors_coordinates['a6502']['x']+anchors_coordinates['a6504']['x'])
        m22 = 2*(-anchors_coordinates['a6502']['y']+anchors_coordinates['a6504']['y'])
        m31 = 2*(-anchors_coordinates['a6503']['x']+anchors_coordinates['a6504']['x'])
        m32 = 2*(-anchors_coordinates['a6503']['y']+anchors_coordinates['a6504']['y'])
        
        B = np.array([[b1],
            [b2],
            [b3]])
        M = np.array([[m11, m12],
            [m21, m22],
            [m31, m32]])
        
        # Perform multilateration with LS
        p = (np.linalg.pinv(M.transpose() @ M)) @ M.transpose() @ B 
        
        # Extract estimated X and Y
        Xest = p[0][0] 
        Yest = p[1][0] 
        
        if 'InitialTime' in mean_data['a6501']: # # Determine real coordinates from mean_data
            Xreal = mean_data['a6501'].iloc[i,1]
            Yreal = mean_data['a6501'].iloc[i,2]
        else:
            Xreal = mean_data['a6501'].iloc[i,0]
            Yreal = mean_data['a6501'].iloc[i,1]
            
        new_row = [Xreal, Yreal, Xest, Yest]
        df_MLT.loc[len(df_MLT)] = new_row # Append the new row with real and estimated coordinates to df_MLT

    return df_MLT

# Funcion to obtain position by AoA+RSSI
def trigonometry(mean_data, config, df_posMLT):
    
    anchors_coordinates = {}
    for anchor in config['anchors']: # getting the data for each anchors
        anchor_id = anchor['id']
        x, y, z = anchor['coordinates']  # get coordinates
        xr, xy, zr = anchor['ref_coordinates']  # get references coordinates
        anchors_coordinates[anchor_id] = {
            'x': x,
            'y': y,
            'z': z,
            'alpha': anchor['alpha'],
            'ref_coordinates': anchor['ref_coordinates']
        }   
    
    df_trigonometry= pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) # Create dataframe o save the results
    
    posTrigonometry = {anchor_id: {'x': [], 'y': []} for anchor_id in ['a6501', 'a6502', 'a6503', 'a6504']} #dict to save the result of each anchor
    for anchor_id in ['a6501', 'a6502', 'a6503', 'a6504']:
        #Calulate the distance from anchors to the position obtained by multilateration
        mean_data[f'{anchor_id}']['Dest_MLT'] = np.sqrt((df_posMLT["Xest"] - anchors_coordinates[f'{anchor_id}']['x'])**2 + (df_posMLT["Yest"] - anchors_coordinates[f'{anchor_id}']['y'])**2 + (anchors_coordinates[f'{anchor_id}']['ref_coordinates'][2] - anchors_coordinates[f'{anchor_id}']['z'])**2)

        # Given the orientation of each anchor, each one has a different equation to calculate, as they need to be calculated considering the same reference axis.
        if anchor_id == 'a6501':
            Xest = abs(-anchors_coordinates[f'{anchor_id}']['x']+mean_data[f'{anchor_id}']['Dest_MLT']*np.sin((np.deg2rad(90-mean_data[f'{anchor_id}']['Azim']))))
            Yest = abs(anchors_coordinates[f'{anchor_id}']['y']-mean_data[f'{anchor_id}']['Dest_MLT']*np.cos((np.deg2rad(90-mean_data[f'{anchor_id}']['Azim']))))
        elif anchor_id == 'a6502':
            Xest = abs(anchors_coordinates[f'{anchor_id}']['x']+mean_data[f'{anchor_id}']['Dest_MLT']*np.cos((np.deg2rad(90-mean_data[f'{anchor_id}']['Azim']))))
            Yest = abs(anchors_coordinates[f'{anchor_id}']['y']-mean_data[f'{anchor_id}']['Dest_MLT']*np.sin((np.deg2rad(90-mean_data[f'{anchor_id}']['Azim']))))
        elif anchor_id == 'a6503':
            Xest = abs(-anchors_coordinates[f'{anchor_id}']['x']+mean_data[f'{anchor_id}']['Dest_MLT']*np.sin((np.deg2rad(90+mean_data[f'{anchor_id}']['Azim']))))
            Yest = abs(anchors_coordinates[f'{anchor_id}']['y']-mean_data[f'{anchor_id}']['Dest_MLT']*np.cos((np.deg2rad(90+mean_data[f'{anchor_id}']['Azim']))))
        else:
            Xest = abs(anchors_coordinates[f'{anchor_id}']['x']+mean_data[f'{anchor_id}']['Dest_MLT']*np.cos((np.deg2rad(90+mean_data[f'{anchor_id}']['Azim']))))
            Yest = abs(anchors_coordinates[f'{anchor_id}']['y']-mean_data[f'{anchor_id}']['Dest_MLT']*np.sin((np.deg2rad(90+mean_data[f'{anchor_id}']['Azim']))))
        
        # Store the Xest and Yest values for each anchor
        posTrigonometry[anchor_id]['x'].append(Xest)
        posTrigonometry[anchor_id]['y'].append(Yest)
            
    # Now, calculate the mean of Xest and Yest for all anchors at each row
    mean_pos = {'Xest_mean': [], 'Yest_mean': []}   
    # Loop through the rows and calculate the mean Xest and Yest across all anchors
    for i in range(len(df_posMLT)):
        
        # Get the Xest and Yest values for each anchor at the current row
        x_values = [posTrigonometry[anchor_id]['x'][0][i] for anchor_id in posTrigonometry]
        y_values = [posTrigonometry[anchor_id]['y'][0][i] for anchor_id in posTrigonometry]

        # Calculate the mean of Xest and Yest for all 4 anchors at this row
        Xest_mean = np.mean(x_values)
        Yest_mean = np.mean(y_values)

        # Append the mean values to the `mean_pos` dictionary
        mean_pos['Xest_mean'].append(Xest_mean)
        mean_pos['Yest_mean'].append(Yest_mean)
  
    # Convert mean_pos dictionary to a DataFrame
    df_mean_pos = pd.DataFrame(mean_pos)
    
    #attribute results in df_trigonometry
    df_trigonometry['Xest'] = df_mean_pos['Xest_mean']
    df_trigonometry['Yest'] = df_mean_pos['Yest_mean']
    df_trigonometry['Xreal'] = df_posMLT['Xreal']
    df_trigonometry['Yreal'] = df_posMLT['Yreal']
    
    return df_trigonometry

def triangulation(mean_data, config):
    
    anchors_coordinates = {}
    for anchor in config['anchors']: # getting the data for each anchors
        anchor_id = anchor['id']
        x, y, z = anchor['coordinates']  # get coordinates
        xr, xy, zr = anchor['ref_coordinates']  # get references coordinates
        anchors_coordinates[anchor_id] = {
            'x': x,
            'y': y,
            'z': z,
            'alpha': anchor['alpha'],
            'ref_coordinates': anchor['ref_coordinates']
        }   
    
    df_triangulation = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) # Create dataframe o save the results
    
    df_triangulation['Xreal'] = mean_data['a6501']['Xreal'] #Assigned real positions
    df_triangulation['Yreal'] = mean_data['a6501']['Yreal']
    
    for i in range(len(mean_data['a6501'])): # Loop through measurements
        
        aoa1 = math.pi - math.radians(mean_data['a6501'].iloc[i,4]) #azimuth 6501
        aoa2 = math.pi/2 - math.radians(mean_data['a6502'].iloc[i,4]) #azimuth 6502
        aoa3 = math.pi - math.radians(mean_data['a6503'].iloc[i,4]) #azimuth 6503
        aoa4 = math.pi/2 - math.radians(mean_data['a6504'].iloc[i,4]) #azimuth 6504
        
        '''
        The article by Ottoy and Kupper was used as a reference for calculating the triangulation, using the least squares estimation.
        '''
        h11 = -math.tan(aoa1)
        h21 = -math.tan(aoa2)
        h31 = -math.tan(aoa3)
        h41 = -math.tan(aoa4)
        h12 = 1
        h22 = 1
        h32 = 1
        h42 = 1

        H = np.array([[h11, h12],
                    [h21, h22],
                    [h31, h32],
                    [h41, h42]])

        c11 = anchors_coordinates['a6501']['y'] - anchors_coordinates['a6501']['x']*math.tan(aoa1)
        c21 = anchors_coordinates['a6502']['y'] - anchors_coordinates['a6502']['x']*math.tan(aoa2)
        c31 = anchors_coordinates['a6503']['y'] - anchors_coordinates['a6503']['x']*math.tan(aoa3)
        c41 = anchors_coordinates['a6504']['y'] - anchors_coordinates['a6504']['x']*math.tan(aoa4)

        c = np.array([[c11],
                    [c21],
                    [c31],
                    [c41]])

        e = np.linalg.inv(H.transpose().dot(H)).dot(H.transpose()).dot(c) #calculo da posição X,Y

        Xest = e[0][0] #isola X
        Yest = e[1][0] #isola Y
        df_triangulation.iloc[i,2] = Xest
        df_triangulation.iloc[i,3] = Yest
    
    return df_triangulation