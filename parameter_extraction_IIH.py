# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:00:37 2019

@author: mango
"""

"""
We calculate some parameters in python
"""
#%% Configurations
project_path_ws = r'D:\users\getang\IIH'
project_path_pc = r'/Users/getang/Documents/EarthResearch/IIH/data'
filename_volume = 'volumetrics_forall_IIH_25112023.csv'
filename_dists = 'allFidDistances_20231127.csv'
Metrics_name = 'IIH_Metrics'
save = True

#%%
import sys
import os
# sys.path.append(os.path.join(project_path_ws, 'Slicertools'))
sys.path.append(r"/Users/getang/Documents/SpaceResearch/spaceflight_associated_neuroocular_syndrome/SANS/src/Preprocessing")

import file_search_tool as fs
import pandas as pd 
import numpy as np
import pickle
import re
from pandas import DataFrame
import matplotlib.pyplot as plt

#%% Functions
def open_pickle(filename:str):
    infile = open(filename,'rb')
    temp_data = pickle.load(infile)
    infile.close()
    return temp_data

def save_list_to_csv(data, output_filename, header=None):
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        if header:
            csvfile.write(header + '\n')
        for row in data:
            row = str(row)
            row_str = ''.join(row)  # Join the elements together without any separation
            csvfile.write(row_str + '\n')  # Write the joined string to the CSV file followed by a newline character
    print(f"Data saved to '{output_filename}'.")

def load_vtk_points(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    poly_data = reader.GetOutput()
    return poly_data.GetPoints()

def vtk_points_to_numpy_array(vtk_points):
    num_points = vtk_points.GetNumberOfPoints()
    data = np.zeros((num_points, 3))
    for i in range(num_points):
        point = vtk_points.GetPoint(i)
        data[i, 0] = point[0]
        data[i, 1] = point[1]
        data[i, 2] = point[2]
    return data

def user_confirmation(prompt="Are you sure? (y/n): "):
    while True:
        user_input = input(prompt).lower()
        if user_input == 'y' or user_input == 'yes':
            return True
        elif user_input == 'n' or user_input == 'no':
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

#%%
"""
Get the parameters for later statistics
Connectin all the information together
Running on my PC
"""
# The volume Metrics
volume = pd.read_csv(os.path.join(project_path_pc, filename_volume))
fn_splits = [x.split('_') for x in volume['filename']]
tagsID  = [f'{x[1]}_{x[2]}' for x in fn_splits]
volume['id'] = tagsID

Metrics = DataFrame()
# Parameters for the volume
Metrics['R_on_sheath_with_nerve'] = volume['R_ONS_Labelmapvolume_cm3']
Metrics['L_on_sheath_with_nerve'] = volume['L_ONS_Labelmapvolume_cm3']
Metrics['R_eyeball'] = volume['R_eyeball_Labelmapvolume_cm3']
Metrics['L_eyeball'] = volume['L_eyeball_Labelmapvolume_cm3']
Metrics['R_lens'] = volume['R_lens_Labelmapvolume_cm3']
Metrics['L_lens'] = volume['L_lens_Labelmapvolume_cm3']
Metrics['Pituitary_gland'] = volume['Pituitary_gland_Labelmapvolume_cm3']
Metrics['id'] = volume['id']

# The distance metrics
eyeball_dist = pd.read_csv(os.path.join(project_path_pc, filename_dists))
# Calculate the area of the elipse by the height and width : Pi * (a/2) * (b/2)
eyeball_dist['R_optcanal_size'] = np.pi * eyeball_dist['h1_R'] * eyeball_dist['w4_R'] / 4
eyeball_dist['L_optcanal_size'] = np.pi * eyeball_dist['h1_L'] * eyeball_dist['w4_L'] / 4

# Merge all the Metrics
Metric4Merge = Metrics.copy() 
Metric4Merge.set_index('id', drop=False, inplace = True)
Eyeball_dist4Merge = eyeball_dist.copy()
Eyeball_dist4Merge.set_index('id', inplace = True)

# Merge the Metrics
MergedMetrics = pd.concat([Metric4Merge, Eyeball_dist4Merge], axis = 1)

# Save the Metrics
if save:   
    print(f'Saving the Metrics to the {project_path_pc}')
    MergedMetrics.to_csv(os.path.join(project_path_pc, f'{Metrics_name}.csv'), index = False)
    MergedMetrics.to_excel(os.path.join(project_path_pc, f'{Metrics_name}.xlsx'), index = False)

# %%
