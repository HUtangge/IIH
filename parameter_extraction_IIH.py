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
filename_volume = 'volumetrics4all.csv'
filename_dists = 'allFidDistances_20231127.csv'
filename_centerline = 'ON_CenterlineMetrics4all.csv'
Metrics_name = 'testing_IIH_Metrics'
save = False

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
Put single files together 
Running on Workstation computer
"""
"""
Volume Metrics
"""
pnsummary = fs.osnj(project_path_ws, 'data', 'Rawdata', 'Summary')
fnsummary = 'volumetrics4all.csv'
pnvolume = fs.osnj(project_path_ws, 'data', 'Rawdata')
fnvolume = r'VolumeMetrics_*.csv'
fl = fs.locateFilesDf(fnvolume, pnvolume, level=2)
for idx,file in enumerate(fl['ff']):
    if idx == 0:
        VolumeMetrics = pd.read_csv(file)
        VolumeMetrics['filename'] = fl['fn_root'][idx]
    else:
        VolumeMetrics_single = pd.read_csv(file)
        VolumeMetrics_single['filename'] = fl['fn_root'][idx]
        VolumeMetrics = VolumeMetrics.append(VolumeMetrics_single, ignore_index = True)

VolumeMetrics['modality'] = VolumeMetrics['filename'].str.extract(r'(T\d)')
VolumeMetrics['fn_root'] = VolumeMetrics['filename'].str.extract(r'(sub-\d+_ses-\d+_T\dw)')

if save:
    VolumeMetrics.to_csv(os.path.join(pnsummary, fnsummary), index = False)

"""
Centerline Metrics
"""
regions = ['R_ON', 'L_ON']
pnsummary = fs.osnj(project_path_ws, 'data', 'Rawdata', 'Summary')
fnsummary = 'ON_CenterlineMetrics4all.csv'
pncenterline = fs.osnj(project_path_ws, 'data', 'Rawdata')
for region in regions:
    fncenterline = f'{region}_CenterlineMetrics*.csv'
    fl = fs.locateFilesDf(fncenterline, pncenterline, level=2)
    for idx,file in enumerate(fl['ff']):
        if (idx == 0) & (region == 'R_ON'):
            CenterlineMetrics = pd.read_csv(file)
            CenterlineMetrics['filename'] = fl['fn_root'][idx]
        else:
            CenterlineMetrics_single = pd.read_csv(file)
            CenterlineMetrics_single['filename'] = fl['fn_root'][idx]
            CenterlineMetrics = CenterlineMetrics.append(CenterlineMetrics_single, ignore_index = True)

CenterlineMetrics['which_eye'] = CenterlineMetrics['filename'].str.extract(r'(^\w_\w{2})')
CenterlineMetrics['fn_root'] = CenterlineMetrics['filename'].str.extract(r'(sub-\d+_ses-\d+_T\dw)')

CenterlineMetrics_wide = CenterlineMetrics.pivot(index='fn_root', columns='which_eye', 
                                                 values=['Radius', 'Length', 'Curvature', 'Torsion', 'Tortuosity',
                                                         'StartPointPosition:0', 'StartPointPosition:1', 'StartPointPosition:2',
                                                         'EndPointPosition:0', 'EndPointPosition:1', 'EndPointPosition:2'])
# Reset the index to make 'filename' a column again
CenterlineMetrics_wide = CenterlineMetrics_wide.reset_index()
# Rename the columns if needed
new_namelist = []
for first, second in list(CenterlineMetrics_wide.columns):
    new_namelist.append(f"{second}_{first}")

CenterlineMetrics_wide.columns=new_namelist
CenterlineMetrics_wide.rename(columns={'_fn_root': 'fn_root'}, inplace=True)

if save:
    CenterlineMetrics_wide.to_csv(os.path.join(pnsummary, fnsummary), index = False)

#%%
"""
Get the parameters for later statistics
Connectin all the information together
Running on my PC
"""
# The volume Metrics
volume = pd.read_csv(os.path.join(project_path_pc, filename_volume))
volume['fn_root'] = volume['filename'].str.extract(r'(sub-\d+_ses-\d+_T\dw)')
centerline = pd.read_csv(os.path.join(project_path_pc, filename_centerline))

# Merge all the parameters
merged_df = pd.merge(volume, centerline, on='filename', how='inner')
merged_df = pd.merge(merged_df, eyeball_dist, on='filename', how='inner')
fn_splits = [x.split('_') for x in merged_df['filename']]
tagsID  = [f'{x[1]}_{x[2]}' for x in fn_splits]
merged_df['id'] = tagsID

Metrics = DataFrame()
# Identity parameters
Metrics['id'] = merged_df['id']
Metrics['modality'] = merged_df['modality']
Metrics['fn_root'] = merged_df['fn_root']

# Parameters for the volume
Metrics['R_on_sheath_with_nerve'] = merged_df['R_ONS_Labelmapvolume_cm3']
Metrics['L_on_sheath_with_nerve'] = merged_df['L_ONS_Labelmapvolume_cm3']
Metrics['R_eyeball'] = merged_df['R_eyeball_Labelmapvolume_cm3']
Metrics['L_eyeball'] = merged_df['L_eyeball_Labelmapvolume_cm3']
Metrics['R_lens'] = merged_df['R_lens_Labelmapvolume_cm3']
Metrics['L_lens'] = merged_df['L_lens_Labelmapvolume_cm3']
Metrics['Pituitary_gland'] = merged_df['Pituitary_gland_Labelmapvolume_cm3']

# The centerline metrics
Metrics['R_on_rad'] = merged_df['R_ON_Radius']
Metrics['R_on_len'] = merged_df['R_ON_Length']
Metrics['R_on_Curvature'] = merged_df['R_ON_Curvature']
Metrics['R_on_Torsion'] = merged_df['R_ON_Torsion']
Metrics['R_on_Tortuosity'] = merged_df['R_ON_Tortuosity']
Metrics['L_on_rad'] = merged_df['L_ON_Radius']
Metrics['L_on_len'] = merged_df['L_ON_Length']
Metrics['L_on_Curvature'] = merged_df['L_ON_Curvature']
Metrics['L_on_Torsion'] = merged_df['L_ON_Torsion']
Metrics['L_on_Tortuosity'] = merged_df['L_ON_Tortuosity']

#%% Merge all the Metrics
# The distance metrics
eyeball_dist = pd.read_csv(os.path.join(project_path_pc, filename_dists))
# Can be done in the parameter extraction
# Calculate the area of the elipse by the height and width : Pi * (a/2) * (b/2)
eyeball_dist['R_optcanal_size'] = np.pi * eyeball_dist['h1_R'] * eyeball_dist['w4_R'] / 4
eyeball_dist['L_optcanal_size'] = np.pi * eyeball_dist['h1_L'] * eyeball_dist['w4_L'] / 4

Metric4Merge = Metrics.copy()  
Eyeball_dist4Merge = eyeball_dist.copy()
Metric4Merge.set_index('names4merge', inplace = True)
Eyeball_dist4Merge.set_index('fn_root', inplace = True)

MergedMetrics = pd.concat([Metric4Merge, Eyeball_dist4Merge], axis = 1)
MergedMetrics.reset_index(inplace=True)
MergedMetrics.drop('index', axis = 1, inplace=True)

# Save the Metrics
if save:   
    print(f'Saving the Metrics to the {project_path_pc}')
    MergedMetrics.to_csv(os.path.join(project_path_pc, f'{Metrics_name}.csv'), index = False)
    MergedMetrics.to_excel(os.path.join(project_path_pc, f'{Metrics_name}.xlsx'), index = False)


# %% Testing code

df_metrics = pd.read_csv('/Users/getang/Documents/EarthResearch/IIH/data/IIH_Metrics.csv')

volume_T1 = pd.read_csv('/Users/getang/Documents/EarthResearch/IIH/data/IIH_volumetrics_T1w.csv')
# %%
