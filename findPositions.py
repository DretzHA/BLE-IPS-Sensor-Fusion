import numpy as np
import pandas as pd
import dataProcessing as dP

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