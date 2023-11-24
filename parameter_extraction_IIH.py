# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:00:37 2019

@author: mango
"""

"""
We calculate some parameters in python
"""
#%% Configurations
# project_path = r'/Users/getang/Documents/SpaceResearch/spaceflight_associated_neuroocular_syndrome/SANS'
project_path_ws = r'D:\users\getang\SANS'
project_path_pc = r'/Users/getang/Documents/SpaceResearch/spaceflight_associated_neuroocular_syndrome/SANS/data'
save = True
testing = False
removing_files = False

#%%
import sys
import os
sys.path.append(os.path.join(project_path_ws, 'Slicertools'))
sys.path.append(r"/Users/getang/Documents/SpaceResearch/spaceflight_associated_neuroocular_syndrome/SANS/src/Preprocessing")

import file_search_tool as fs
import pandas as pd 
import numpy as np
import pickle
import re
import csv
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
# subject information

#%%
"""
Put single files together 
Workstation computer
"""
region = 'R_optic_nerve'
pnsummary = fs.osnj(project_path_ws, 'Summary')
fnsummary = '06052021_optimal_FeretdiameterMetrics4all.csv'
pnferetdiameter = fs.osnj(project_path_ws, 'Cosmonaut_BIDS')
# fnferetdiameter = f'{region}_optimal_Diametermodel*.vtk'
fnferetdiameter = 'FeretdiameterMetrics_optimal*.csv' 
fl = fs.locateFiles(fnferetdiameter, pnferetdiameter, level=4)
for idx,file in enumerate(fl):
    if idx == 0:
        FeretdiameterMetrics = pd.read_csv(file)
    else:
        FeretdiameterMetrics_single = pd.read_csv(file)
        FeretdiameterMetrics = FeretdiameterMetrics.append(FeretdiameterMetrics_single, ignore_index = True)

FeretdiameterMetrics.to_csv(os.path.join(pnsummary, fnsummary), index = False)

#%% For preparing the filenames to copy from the work station computer
fnFIDSptn = 'fids_*T1w.fcsv' # Cosmo1mm / Astronauts / Cosmonauts
fnSEGSptn = 'Segs_*.seg.nrrd' # Way to use: name = fnSEGSptn %(a,b)
pnMRIptn = os.path.join(project_path, 'Raw_data', 'Cosmonaut_BIDS')
ptnMRI = 'sub*_T1w.nii'
fl = fs.locateFiles(ptnMRI, pnMRIptn, level=3)
header = 'filename'
filename = f'filenames_for_sweden_age_prediction.csv' # f'Cosmonauts_shortened_filenames_polar_projection_{region}.csv'
save_list_to_csv(fl, fs.osnj(project_path_ws, 'Raw_data', 'Summary', filename), header=header)


"""
Connect all the Right eye files together on the workstation computer
This is a 4D matrix with 3D mapping and the 4th dimension is the files
"""
region = 'L_eyeball'
degree_to_center = 90
pnsummary = fs.osnj(project_path_ws, 'Raw_data', 'Summary')
fnsummary = f'26042023_{region}_polar_projectionmap4all.csv'
pnpolarprojection = fs.osnj(project_path_ws, 'Raw_data', 'Cosmonaut_BIDS')
fnpolarprojection = f'Polar_projection_{region}_{degree_to_center}*.vtp' 
fl = fs.locateFiles(fnpolarprojection, pnpolarprojection, tailOnly = False, level=4)
header = 'filename'
filename = f'filenames_polar_projection_{region}_{degree_to_center}.csv' # f'Cosmonauts_shortened_filenames_polar_projection_{region}.csv'
save_list_to_csv(fl, fs.osnj(project_path_ws, 'Raw_data', 'Summary', filename), header=header)

for idx,file in enumerate(fl):
    print(file)
    if idx == 0:
        PolarprojectionMetrics = vtk_points_to_numpy_array(load_vtk_points(file))
        PolarprojectionMetrics = np.expand_dims(PolarprojectionMetrics, axis=-1)
    else:
        PolarprojectionMetric = vtk_points_to_numpy_array(load_vtk_points(file))
        PolarprojectionMetric = np.expand_dims(PolarprojectionMetric, axis=-1)
        PolarprojectionMetrics = np.concatenate((PolarprojectionMetrics, PolarprojectionMetric), axis=2)

if save:
    PolarprojectionMetrics.to_csv(os.path.join(pnsummary, fnsummary), index = False)

if removing_files:
    print(fl)
    if user_confirmation("Do you really want to delete these files? (y/n): "):
        print("Deleting...")
        # Add your code to execute upon confirmation here
        for idx,file in enumerate(fl):
            print(file)
            if os.path.isfile(file):
                print(file)
                os.remove(file)
            else:
                warning.warn(f"The file {file} does not exist.")
    else:
        print("Cancelled deleting files.")

"""
This is for connecting centerline Metrics, we need to rename some columns, So I run 
it separately
"""
region = 'R_optic_nerve'
pnsummary = fs.osnj(project_path_ws, 'Summary')
fnsummary = '21042021_R_optic_nerve_CenterlineMetrics4all.csv'
pncenterline = fs.osnj(project_path_ws, 'Cosmonaut_BIDS')
fncenterline = f'{region}_CenterlineMetrics*.csv'
fl = fs.locateFiles(fncenterline, pncenterline, level=4)
for idx,file in enumerate(fl):
    if idx == 0:
        CenterlineMetrics = pd.read_csv(file)
    else:
        CenterlineMetrics_single = pd.read_csv(file)
        CenterlineMetrics = CenterlineMetrics.append(CenterlineMetrics_single, ignore_index = True)

column_names = list(CenterlineMetrics.columns)
columns = dict((column_name, f'{region}_{column_name}') for column_name in column_names)
CenterlineMetrics.rename(columns=columns, inplace = True)
CenterlineMetrics.to_csv(os.path.join(pnsummary, fnsummary), index = False)

"""
TangGe: Add the columns for the subject and pre- post- measurement
"""
# !!! Attention Run on workstation computer
import pickle
from pandas import DataFrame
fnMRI = r'D:\SANS\Raw_data\Astronaut_BIDS'
ptnMRI = 'Denoised_*_T1w.nii'
fl = fs.locateFiles(ptnMRI, fnMRI, level=3, tailOnly=True)
su.save_pickle(fl, 'D://astro_filenames.pickle')

#%%
"""
Get the parameters for later statistics
Running on my PC
"""
save = True
# !!! Attention Run on my laptop
fl = open_pickle(fs.osnj(project_path_pc, 'cosmo_filenames.pickle'))
### Code the subject information
sub_info = np.zeros((len(fl), 2))
group = []
session = []
for idx, ff in enumerate(fl):
    which_sub = re.search('sub-(.+?)_ses', ff)
    if 'control' in which_sub.group(1):
        sub_info[idx, 0] = 100 + int(re.search('\d+', which_sub.group(1)).group(0))
        group.append('control')
    elif 'cosmonaut' in which_sub.group(1):
        sub_info[idx, 0] = 200 + int(re.search('\d+', which_sub.group(1)).group(0))
        group.append('cosmonaut')
    if 'preflight' in ff:
        sub_info[idx, 1] = 1
        # session.append('pre')
    elif 'postflight' in ff:
        # 8 days afterlanding
        sub_info[idx, 1] = 2
        # session.append('post')
    elif 'followup' in ff:
        # half a year after landing
        sub_info[idx, 1] = 3
        # session.append('followup')
group = pd.Series(group, name = 'group')
# session = pd.Series(session, name = 'session')
sub_info = DataFrame(sub_info, columns=['subject', 'session'])
sub_info = pd.concat([group, sub_info], axis=1)
# Metrics = pd.concat([sub_info, data], axis=1)

#%% For astronauts since they have different data structure 
# !!! Attention Run on my laptop
fl = open_pickle(fs.osnj(project_path_pc,'astro_filenames.pickle'))
### Code the subject information
# for consistency, the additional measurement for the astronauts are coded as 4 and 5
sub_info = np.zeros((len(fl), 2))
group = []
session = []
for idx, ff in enumerate(fl):
    which_sub = re.search('(?<=(control|tronaut))([A-Z]|\d+)', ff).group(0)    
    if len(which_sub) == 1:
        sub_info[idx, 0] = 100 + ord(which_sub) - 64
        group.append('control')
    elif len(which_sub) == 2:
        sub_info[idx, 0] = 200 + int(which_sub)
        group.append('astronaut')
    if re.search('(preflight|ses-1)', ff):
        sub_info[idx, 1] = 1
        # session.append('pre')
    elif re.search('(postflight1|ses-2)', ff):
        # 2-4 days after landing
        sub_info[idx, 1] = 2
        # session.append('post')
    elif re.search('(postflight2|ses-3)', ff):
        # 2 weeks after landing
        sub_info[idx, 1] = 4    
    elif re.search('(postflight3)', ff):
        # 2 months after landing
        sub_info[idx, 1] = 5  
    elif re.search('(followup|ses-4)', ff):
        # Half a year after landing
        sub_info[idx, 1] = 3
        # session.append('followup')
group = pd.Series(group, name = 'group')
# session = pd.Series(session, name = 'session')
sub_info = DataFrame(sub_info, columns=['subject', 'session'])
sub_info = pd.concat([group, sub_info], axis=1)
# Metrics = pd.concat([sub_info, data], axis=1)

#%%
"""
Tang Ge : Cosmonauts Part
Connectin all the information together
"""
save = False
fl = open_pickle(fs.osnj(project_path_pc, 'cosmo_filenames.pickle'))

R_centerline = pd.read_csv(os.path.join(project_path_pc, '21042021_R_optic_nerve_CenterlineMetrics4all.csv'))
L_centerline = pd.read_csv(os.path.join(project_path_pc, '21042021_L_optic_nerve_CenterlineMetrics4all.csv'))
volume = pd.read_csv(os.path.join(project_path_pc, '21042021_VolumeMetrics4all.csv'))
feret_2104 = pd.read_csv(os.path.join(project_path_pc, '21042021_FeretdiameterMetrics4all.csv'))
feret_0605 = pd.read_csv(os.path.join(project_path_pc, '06052021_optimal_FeretdiameterMetrics4all.csv'))
R_recomputedcenterline = pd.read_csv(os.path.join(project_path_pc, '15102021_R_optic_nerve_recomputedCenterlineGeometry.csv'))
L_recomputedcenterline = pd.read_csv(os.path.join(project_path_pc, '15102021_L_optic_nerve_recomputedCenterlineGeometry.csv'))
opticalcanal = pd.read_csv(os.path.join(project_path_pc, '25112021_optic_nerve_canalMetrics4all.csv'))

# Create an empty dataframe to store the metrics
Metrics = DataFrame(fl, columns=['filename'])

# Add additional column for later merge with the morphometry data
names4merge = []
for idx, ff in enumerate(fl):
    names4merge.append(ff[9:-4])
Metrics['names4merge'] = names4merge

# Parameters for the volume
Metrics['R_on_sheath'] = volume['R_optic_nerve_sheath_anterior_Labelmapvolume_cm3']
Metrics['L_on_sheath'] = volume['L_optic_nerve_sheath_anterior_Labelmapvolume_cm3']
Metrics['R_eyeball'] = volume['R_eyeball_Labelmapvolume_cm3']
Metrics['L_eyeball'] = volume['L_eyeball_Labelmapvolume_cm3']
Metrics['R_on_sheath_with_nerve'] = volume['R_optic_nerve_sheath_anterior_with_nerve_Labelmapvolume_cm3']
Metrics['L_on_sheath_with_nerve'] = volume['L_optic_nerve_sheath_anterior_with_nerve_Labelmapvolume_cm3']
Metrics['Pituitary_gland'] = volume['Pituitary_gland_Labelmapvolume_cm3']
Metrics['Pituitary_gland_posterior'] = volume['Pituitary_gland_dense_part_Labelmapvolume_cm3']
Metrics['Pituitary_gland_anterior'] = volume['Pituitary_gland_Labelmapvolume_cm3'] - volume['Pituitary_gland_dense_part_Labelmapvolume_cm3']

# parameters for the centerline
Metrics['R_on_rad'] = R_centerline['R_optic_nerve_Radius']
Metrics['R_on_len'] = R_centerline['R_optic_nerve_Length']
Metrics['R_on_Curvature'] = R_centerline['R_optic_nerve_Curvature']
Metrics['R_on_Torsion'] = R_centerline['R_optic_nerve_Torsion']
Metrics['R_on_Tortuosity'] = R_centerline['R_optic_nerve_Tortuosity']
Metrics['L_on_rad'] = L_centerline['L_optic_nerve_Radius']
Metrics['L_on_len'] = L_centerline['L_optic_nerve_Length']
Metrics['L_on_Curvature'] = L_centerline['L_optic_nerve_Curvature']
Metrics['L_on_Torsion'] = L_centerline['L_optic_nerve_Torsion']
Metrics['L_on_Tortuosity'] = L_centerline['L_optic_nerve_Tortuosity']

# Recomputed centerline geometry metrics
Metrics['R_on_mean_curvature'] = R_recomputedcenterline['R_optic_nerve_mean_curvature']
Metrics['R_on_median_curvature'] = R_recomputedcenterline['R_optic_nerve_median_curvature']
Metrics['R_on_max_curvature'] = R_recomputedcenterline['R_optic_nerve_max_curvature']
Metrics['R_on_argmax_curvature'] = R_recomputedcenterline['R_optic_nerve_1thargmax_curvature']
Metrics['R_on_argmax_torsion'] = R_recomputedcenterline['R_optic_nerve_1thargmax_torsion']
Metrics['L_on_mean_curvature'] = L_recomputedcenterline['L_optic_nerve_mean_curvature']
Metrics['L_on_median_curvature'] = L_recomputedcenterline['L_optic_nerve_median_curvature']
Metrics['L_on_max_curvature'] = L_recomputedcenterline['L_optic_nerve_max_curvature']
Metrics['L_on_argmax_curvature'] = L_recomputedcenterline['L_optic_nerve_1thargmax_curvature']
Metrics['L_on_argmax_torsion'] = L_recomputedcenterline['L_optic_nerve_1thargmax_torsion']

# Feret diameter perpendicular to the centerline (3mm/optic nerve and 5mm/optic nerve sheath)
Metrics['R_3mm_maxdia'] = feret_2104['R_optic_nerve_maxferetdiameter']
Metrics['R_3mm_perdia'] = feret_2104['R_optic_nerve_perpendicularferetdiameter']
Metrics['R_3mm_area'] = feret_2104['R_optic_nerve_curveareamm2']
Metrics['R_3mm_len'] = feret_2104['R_optic_nerve_curvelengthmm']
Metrics['L_3mm_maxdia'] = feret_2104['L_optic_nerve_maxferetdiameter']
Metrics['L_3mm_perdia'] = feret_2104['L_optic_nerve_perpendicularferetdiameter']
Metrics['L_3mm_area'] = feret_2104['L_optic_nerve_curveareamm2']
Metrics['L_3mm_len'] = feret_2104['L_optic_nerve_curvelengthmm']
Metrics['R_5mm_maxdia'] = feret_2104['R_optic_nerve_sheath_anterior_with_nerve_maxferetdiameter']
Metrics['R_5mm_perdia'] = feret_2104['R_optic_nerve_sheath_anterior_with_nerve_perpendicularferetdiameter']
Metrics['R_5mm_area'] = feret_2104['R_optic_nerve_sheath_anterior_with_nerve_curveareamm2']
Metrics['R_5mm_len'] = feret_2104['R_optic_nerve_sheath_anterior_with_nerve_curvelengthmm']
Metrics['L_5mm_maxdia'] = feret_2104['L_optic_nerve_sheath_anterior_with_nerve_maxferetdiameter']
Metrics['L_5mm_perdia'] = feret_2104['L_optic_nerve_sheath_anterior_with_nerve_perpendicularferetdiameter']
Metrics['L_5mm_area'] = feret_2104['L_optic_nerve_sheath_anterior_with_nerve_curveareamm2']
Metrics['L_5mm_len'] = feret_2104['L_optic_nerve_sheath_anterior_with_nerve_curvelengthmm']

# Optimal Feret diameter (3mm)
Metrics['R_3mm_maxdia_optimal'] = feret_0605['R_optic_nerve_maxferetdiameter']
Metrics['R_3mm_perdia_optimal'] = feret_0605['R_optic_nerve_perpendicularferetdiameter']
Metrics['R_3mm_area_optimal'] = feret_0605['R_optic_nerve_curveareamm2']
Metrics['R_3mm_len_optimal'] = feret_0605['R_optic_nerve_curvelengthmm']
Metrics['L_3mm_maxdia_optimal'] = feret_0605['L_optic_nerve_maxferetdiameter']
Metrics['L_3mm_perdia_optimal'] = feret_0605['L_optic_nerve_perpendicularferetdiameter']
Metrics['L_3mm_area_optimal'] = feret_0605['L_optic_nerve_curveareamm2']
Metrics['L_3mm_len_optimal'] = feret_0605['L_optic_nerve_curvelengthmm']

# Optical canal
Metrics['R_optcanal_area'] = opticalcanal['R_optcanal_area']
Metrics['R_optcanal_perimeter'] = opticalcanal['R_optcanal_length']
Metrics['L_optcanal_area'] = opticalcanal['L_optcanal_area']
Metrics['L_optcanal_perimeter'] = opticalcanal['L_optcanal_length']

"""
Tang Ge : Merge all the Metrics
"""
eyeball_dist = pd.read_csv(os.path.join(project_path_pc, 'allFidDistances_Cosmo02mm.csv'))
# Can be done in the parameter extraction
# Calculate the area of the elipse by the height and width : Pi * (a/2) * (b/2)
eyeball_dist['R_optcanal_size'] = np.pi * eyeball_dist['h1_R'] * eyeball_dist['w4_R'] / 4
eyeball_dist['L_optcanal_size'] = np.pi * eyeball_dist['h1_L'] * eyeball_dist['w4_L'] / 4
morphometrics = pd.read_excel(os.path.join(project_path_pc, 'TIV_Cosmonauts.xlsx'))
Vcsf_morphometrics = pd.read_csv(os.path.join(project_path_pc, 'Cosmonauts_neuromorphometrics_Vcsf_python.csv'))
columes_for_vcsf = ['names', 'l3thVen', 'r3thVen', 'l4thVen', 'r4thVen', 'lInfLatVen',
                    'lLatVen', 'rInfLatVen', 'rLatVen']

Metric4Merge = Metrics.copy()  
Eyeball_dist4Merge = eyeball_dist.copy()
Morphometrics4Merge = morphometrics.copy()
Vcsf_morphometrics4Merge = Vcsf_morphometrics[columes_for_vcsf]
Vcsf_morphometrics4Merge.insert(3, '3thVen', Vcsf_morphometrics['l3thVen']+Vcsf_morphometrics['r3thVen'])
Vcsf_morphometrics4Merge.insert(6, '4thVen', Vcsf_morphometrics['l4thVen']+Vcsf_morphometrics['r4thVen'])
Vcsf_morphometrics4Merge.insert(9, 'lLVen', Vcsf_morphometrics['lInfLatVen']+Vcsf_morphometrics['lLatVen'])
Vcsf_morphometrics4Merge.insert(12, 'rLVen', Vcsf_morphometrics['rInfLatVen']+Vcsf_morphometrics['rLatVen'])
Vcsf_morphometrics4Merge.insert(13, 'LVen', Vcsf_morphometrics['lInfLatVen']+Vcsf_morphometrics['lLatVen']+Vcsf_morphometrics['rInfLatVen']+Vcsf_morphometrics['rLatVen'])
Vcsf_morphometrics4Merge.insert(14, 'intracranialVen', Vcsf_morphometrics4Merge['3thVen']+Vcsf_morphometrics4Merge['lLVen']+Vcsf_morphometrics4Merge['rLVen'])
Vcsf_morphometrics4Merge.insert(15, 'AllVen', Vcsf_morphometrics4Merge['intracranialVen']+Vcsf_morphometrics4Merge['4thVen'])

# Set the index name as filename for match
# sub_info.set_index('filename', inplace = True)
Metric4Merge.set_index('names4merge', inplace = True)
Eyeball_dist4Merge.set_index('fn_root', inplace = True)
Morphometrics4Merge.set_index('Name', inplace = True)
Vcsf_morphometrics4Merge.set_index('names', inplace = True)

# Merge the Metrics
MergedMetrics = pd.concat([Metric4Merge, Morphometrics4Merge, Vcsf_morphometrics4Merge, Eyeball_dist4Merge], axis = 1)
MergedMetrics.reset_index(inplace=True)
MergedMetrics.drop('index', axis = 1, inplace=True)

# For keep the cosmonauts data only 
MergedMetrics.dropna(subset=['filename'], inplace=True)
MergedMetrics.columns = MergedMetrics.columns.str.replace(' ', '')

# Save the Metrics
if save:   
    MergedMetrics.to_csv(os.path.join(project_path_pc, 'testing_Cosmonauts02mmT1_morphometrics.csv'), index = False)
    MergedMetrics.to_excel(os.path.join(project_path_pc, 'testing_Cosmonauts02mmT1_morphometrics.xlsx'), index = False)

#%%
"""
In progress ...
Tang Ge : Connect the parameters from previous project
"""
# for idx, ff in enumerate(Metrics['filename']):  
#     if idx > 66:
#         if ff != morphometrics['Name'][idx+1]:
#             print(idx)
#             print(ff)
#             break

#%% 
"""
Tang Ge : Astronauts Part
"""
save = False
fl = open_pickle(fs.osnj(project_path_pc,'astro_filenames.pickle'))
# !!! temportary solution
del fl[33:41] 

astro_volume = pd.read_csv(os.path.join(project_path_pc, '17112021_Astro02mm_volumetrics_forall.csv'))

Metrics = DataFrame(fl, columns=['filename'])
# Tang Ge: Instead of using coding of subject information, 
# the identity for each file by filename 
names4merge = []
for idx, ff in enumerate(fl):
    names4merge.append(ff[9:-4])
Metrics['names4merge'] = names4merge

# Astronauts part: separate from cosmonauts
Metrics['R_on_sheath_with_nerve'] = astro_volume['R_optic_nerve_sheath_anterior_with_nerve_Labelmapvolume_cm3']
Metrics['L_on_sheath_with_nerve'] = astro_volume['L_optic_nerve_sheath_anterior_with_nerve_Labelmapvolume_cm3']

# Merge all the parameters
eyeball_dist = pd.read_csv(os.path.join(project_path_pc, 'allFidDistances_AstronautsT1_WithOrbitalRimDistAndOpticalCanal.csv'))
# Can be done in the parameter extraction
# Calculate the area of the elipse by the height and width : Pi * (a/2) * (b/2)
eyeball_dist['R_optcanal_size'] = np.pi * eyeball_dist['h1_R'] * eyeball_dist['w4_R'] / 4
eyeball_dist['L_optcanal_size'] = np.pi * eyeball_dist['h1_L'] * eyeball_dist['w4_L'] / 4
morphometrics = pd.read_excel(os.path.join(project_path_pc, 'TIV_Astronauts.xlsx'))
# Remove the space in the string
morphometrics.columns = morphometrics.columns.str.strip()
morphometrics['Name'] = morphometrics['Name'].str.strip()
Vcsf_morphometrics = pd.read_csv(os.path.join(project_path_pc, 'Astronauts_neuromorphometrics_Vcsf_python.csv'))
columes_for_vcsf = ['names', 'l3thVen', 'r3thVen', 'l4thVen', 'r4thVen', 'lInfLatVen',
                    'lLatVen', 'rInfLatVen', 'rLatVen']

Metric4Merge = Metrics.copy()  
Eyeball_dist4Merge = eyeball_dist.copy()
Morphometrics4Merge = morphometrics.copy()
Vcsf_morphometrics4Merge = Vcsf_morphometrics[columes_for_vcsf]
Vcsf_morphometrics4Merge.insert(3, '3thVen', Vcsf_morphometrics['l3thVen']+Vcsf_morphometrics['r3thVen'])
Vcsf_morphometrics4Merge.insert(6, '4thVen', Vcsf_morphometrics['l4thVen']+Vcsf_morphometrics['r4thVen'])
Vcsf_morphometrics4Merge.insert(9, 'lLVen', Vcsf_morphometrics['lInfLatVen']+Vcsf_morphometrics['lLatVen'])
Vcsf_morphometrics4Merge.insert(12, 'rLVen', Vcsf_morphometrics['rInfLatVen']+Vcsf_morphometrics['rLatVen'])
Vcsf_morphometrics4Merge.insert(13, 'intracranialVen', Vcsf_morphometrics4Merge['3thVen']+Vcsf_morphometrics4Merge['lLVen']+Vcsf_morphometrics4Merge['rLVen'])
Vcsf_morphometrics4Merge.insert(14, 'AllVen', Vcsf_morphometrics4Merge['intracranialVen']+Vcsf_morphometrics4Merge['4thVen'])

# Set the index name as filename for matching
Metric4Merge = Metric4Merge.set_index('names4merge')
Eyeball_dist4Merge = Eyeball_dist4Merge.set_index('fn_root')
Morphometrics4Merge = Morphometrics4Merge.set_index('Name')
Vcsf_morphometrics4Merge = Vcsf_morphometrics4Merge.set_index('names')

# Merge the Metrics
MergedMetrics = pd.concat([Metric4Merge, Morphometrics4Merge, Vcsf_morphometrics4Merge, Eyeball_dist4Merge], axis = 1)
MergedMetrics.reset_index(inplace = True)
MergedMetrics.drop('index', axis = 1, inplace=True)
# MergedMetrics.drop(list(range(158,256)), inplace=True)

# For keep the cosmonauts data only 
MergedMetrics.dropna(subset=['filename'], inplace=True)
MergedMetrics.columns = MergedMetrics.columns.str.replace(' ', '')

# MergedMetrics.to_csv('/Users/getang/Documents/Summary/Mergedmetrics.csv', index_label = 'Names')
if save:
    MergedMetrics.to_csv(os.path.join(project_path_pc, 'testing_Astronauts02mmT1_morphometrics.csv'), index = False)
    MergedMetrics.to_excel(os.path.join(project_path_pc, 'Testing_Astronauts02mmT1_morphometrics.xlsx'), index = False)

#%%

# Merge the Metrics
MergedMetrics = pd.concat([sub_info, Morphometrics4Merge, Vcsf_morphometrics4Merge, Eyeball_dist4Merge, Metric4Merge], axis = 1)
MergedMetrics.reset_index(inplace=True)
MergedMetrics.rename(columns={"index": "Names"}, inplace=True)
MergedMetrics.drop(list(range(158,256)), inplace=True)

# MergedMetrics.to_csv('/Users/getang/Documents/Summary/Mergedmetrics.csv', index_label = 'Names')
MergedMetrics.to_csv(os.path.join(project_path_pc, 'Cosmonauts02mmT1_morphometrics.csv'), index = False)
MergedMetrics.to_excel(os.path.join(project_path_pc, 'Cosmonauts02mmT1_morphometrics.xlsx'), index = False)

#%% 
"""
Reorganize the result data from R 
Thus the two triangles present the correlation and the probability separately
"""
result_path = '/Users/getang/Documents/LMU/SANS/results/tables'
decimals_to_keep = 5
def get_lowertriangular(rdm):
    """
    Function from NMA : 
    https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/fMRI/load_cichy_fMRI_MEG.ipynb#scrollTo=vt8ai7Mmr9_Q
    returns lower triangular part of the matrix
    """
    num_conditions = rdm.shape[0]
    return rdm[np.tril_indices(num_conditions,-1)]

correlation_matrix = pd.read_csv(os.path.join(result_path, 'correlation_table.csv'), index_col=0)
correlation_probability_matrix = pd.read_csv(os.path.join(result_path, 'correlation_probability_table.csv'), index_col=0)
colume_names = list(correlation_matrix.columns)

correlation_matrix = correlation_matrix.to_numpy()
correlation_probability_matrix = correlation_probability_matrix.to_numpy()
correlation_with_probability = correlation_probability_matrix.copy()

correlation_probability = get_lowertriangular(correlation_matrix)
correlation_with_probability[np.tril_indices(correlation_matrix.shape[0],-1)] = correlation_probability
correlation_with_probability = np.round(correlation_with_probability, decimals = decimals_to_keep)
correlation_with_probability = pd.DataFrame(correlation_with_probability, colume_names, colume_names)

correlation_with_probability.to_csv(os.path.join(result_path, 'correlation_table_with_probability.csv'), index = True, sep=';')

correlation_with_probability = pd.read_csv(os.path.join(result_path, 'correlation_table_with_probability.csv'), index_col=0)

#%%
"""
TangGe : Because before 93, the diameter plane is not checked, so I did this for the
models has 2 lines
"""
region = 'L_optic_nerve'
fndiametermodel = fs.osnj(project_path_ws, 'Cosmonaut_BIDS')
pndiametermodel = f'{region}_optimal_Diametermodel*.vtk' 
fl = fs.locateFiles(pndiametermodel, fndiametermodel, level=4)
num_of_polylines = np.zeros(len(fl))
for idx, ff in enumerate(fl):
    ff = ff.replace('/','\\')
    success, diametermodel = loadModel(ff, returnNode=True)
    num_of_polylines[idx] = diametermodel.GetPolyData().GetNumberOfLines()
    slicer.mrmlScene.RemoveNode(diametermodel)
su.save_pickle(num_of_polylines, 'D://GeTang//L_num.pickle')

#%% 
"""
Recompute the centerline geometry : 
    mean_curvature, median_curvature, max_curvature, 1thargmax_curvature, 2thargmax_curvature
    3thargmax_curvature, argmax_torsion, 
"""
def geoCenterline(curve):
    """
    The source of Equation
    https://en.wikipedia.org/wiki/Torsion_of_a_curve
    https://en.wikipedia.org/wiki/Curvature#Space_curves
    
    Parameters
    ----------
    curve : nparray
        All the points on the centerline, the same diameter plane. n by 3 np array

    Returns
    -------
    Curvature : n by 1 array
    Torsion : n by 1 array

    """
    dx = curve[:,0]
    dy = curve[:,1]
    dz = curve[:,2]
    dx_dt = np.gradient(dx)
    dy_dt = np.gradient(dy)
    dz_dt = np.gradient(dz)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    d2z_dt2 = np.gradient(dz_dt)
    d3x_dt3 = np.gradient(d2x_dt2)
    d3y_dt3 = np.gradient(d2y_dt2)
    d3z_dt3 = np.gradient(d2z_dt2)
    
    curvature = (np.sqrt((d2x_dt2 * dy_dt - dx_dt * d2y_dt2)**2 + 
                        (d2x_dt2 * dz_dt - dx_dt * d2z_dt2)**2 + 
                        (d2y_dt2 * dz_dt - dy_dt * d2z_dt2)**2) / 
                 (dx_dt * dx_dt + dy_dt * dy_dt + dz_dt * dz_dt)**1.5)
    
    torsion = (((dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) * d3z_dt3 + 
                        (d2x_dt2 * dz_dt - dx_dt * d2z_dt2) * d3y_dt3 + 
                        (dy_dt * d2z_dt2 - d2y_dt2 * dz_dt) * d3x_dt3) 
               / ((d2x_dt2 * dy_dt - dx_dt * d2y_dt2)**2 + 
                        (d2x_dt2 * dz_dt - dx_dt * d2z_dt2)**2 + 
                        (d2y_dt2 * dz_dt - dy_dt * d2z_dt2)**2))
    return curvature, torsion 

# Put all the subjects data together
region = 'L_optic_nerve'
pnsummary = fs.osnj(project_path_ws, 'Summary')
fnsummary = '15102021_recomputedCenterlineGeometry4all.csv'
pncenterlinearray = fs.osnj(project_path_ws, 'Cosmonaut_BIDS')
fncenterlinearray = f'{region}_resampledCenterlinearray*.pickle' 
#mean_curvature, median_curvature, max_curvature, 1thargmax_curvature, 2thargmax_curvature
 #   3thargmax_curvature, argmax_torsion, 
fl = fs.locateFiles(fncenterlinearray, pncenterlinearray, level=4)
recomputedCenterlineGeometry = {}
recomputedCenterlineGeometry[f'{region}_mean_curvature'] = []
recomputedCenterlineGeometry[f'{region}_median_curvature'] = []
recomputedCenterlineGeometry[f'{region}_max_curvature'] = []
recomputedCenterlineGeometry[f'{region}_1thargmax_curvature'] = []
recomputedCenterlineGeometry[f'{region}_1thargmax_torsion'] = []
curvature_all = np.zeros((158, 35))
for idx,file in enumerate(fl):
    print(file)    
    curve = open_pickle(file)
    controlpoints = np.where(curve[:,3] == 1)
    curve = curve[:,:3]
    curvature, torsion = geoCenterline(curve)
    tmp_curvature = curvature[controlpoints]
    plt.plot(tmp_curvature)
    recomputedCenterlineGeometry[f'{region}_mean_curvature'].append(np.mean(tmp_curvature))
    recomputedCenterlineGeometry[f'{region}_median_curvature'].append(np.median(tmp_curvature))
    recomputedCenterlineGeometry[f'{region}_max_curvature'].append(np.max(tmp_curvature))
    recomputedCenterlineGeometry[f'{region}_1thargmax_curvature'].append(np.argmax(tmp_curvature))
    recomputedCenterlineGeometry[f'{region}_1thargmax_torsion'].append(np.argmax(torsion))
    curvature_all[idx,:len(tmp_curvature)] = tmp_curvature

curvature_all[curvature_all == 0] = np.nan
save_pickle(curvature_all, fs.osnj(project_path_ws, 'Cosmonaut_BIDS\Summary\15102021_L_optic_nerve_curvature_4all.pickle'), index = False)
recomputedCenterlineGeometry = pd.DataFrame(recomputedCenterlineGeometry)
recomputedCenterlineGeometry.to_csv(fs.osnj(project_path_ws, 'Summary\15102021_L_optic_nerve_recomputedCenterlineGeometry.csv', index = False)

    
#%%
"""
Extract parameters from Ahmadi
"""
# Some help functions for this part
def readSlicerAnnotationFiducials(ff):
    fids = pd.read_csv(ff,
                       comment='#',
                       header=None,
                       names=['id','x','y','z','ow','ox','oy','oz','vis','sel','lock','label','desc','associatedNodeID'],
                       engine='python')
    return fids

def df_dist(df,pt1,pt2):
    p1 = df.loc[pt1,['x','y','z']].values.astype(float)
    p2 = df.loc[pt2,['x','y','z']].values.astype(float)
    return np.linalg.norm(p1-p2)

def fid_measures(df, withEyeOrbDist=True):
    #df = df.set_index('label')
    d = dict()
    for side in ['L','R']: 
        d['d1_%s'%side] = df_dist(df,'individualized_center_%s_lens'%side,'individualized_center_%s_eyeball'%side)
        d['d2_%s'%side] = df_dist(df,'individualized_center_%s_eyeball'%side,'nerve_tip_%s'%side)
        d['d3_%s'%side] = df_dist(df,'individualized_center_%s_lens'%side,'eyeball_back_%s'%side)
        d['w1_%s'%side] = df_dist(df,'eyeball_midline_%s_lat'%side,'eyeball_midline_%s_med'%side)
        d['w2_%s'%side] = df_dist(df,'nerve_baseline_muscle_%s_lat'%side,'nerve_baseline_muscle_%s_med'%side)
        d['w3_%s'%side] = df_dist(df,'nerve_baseline_bone_%s_lat'%side,'nerve_baseline_bone_%s_med'%side)
        d['n1_%s'%side] = df_dist(df,'nerve_width_%s_lat'%side,'nerve_width_%s_med'%side)
        d['h1_%s'%side] = df_dist(df,'optcanal_height_%s_inf'%side,'optcanal_height_%s_sup'%side)
        d['w4_%s'%side] = df_dist(df,'optcanal_width_%s_lat'%side,'optcanal_width_%s_med'%side)
        
        # estimate lsq-plane of orbital rim
        if withEyeOrbDist:
            ptsOrb = df.loc[['orbital_rim_%s_lat'%side,
                              'orbital_rim_%s_med'%side,
                              'orbital_rim_%s_sup'%side,
                              'orbital_rim_%s_inf'%side],['x','y','z']].values.astype(float)
            normal,offset,R = lstsqPlaneEstimation(ptsOrb)
            ptsOrbMean = np.mean(ptsOrb,axis=0)
            # Eyecenter to the plane
            ptsEyeCtr  = df.loc['individualized_center_%s_eyeball'%side,['x','y','z']].values.astype(float)
            disteye = np.dot(normal,ptsEyeCtr-ptsOrbMean)
            d['d4_%s'%side] = disteye
            # Lens center to the plane
            ptsLensCtr  = df.loc['individualized_center_%s_lens'%side,['x','y','z']].values.astype(float)
            distlens = np.dot(normal,ptsLensCtr-ptsOrbMean)
            d['d5_%s'%side] = distlens
    return d

def vol_nerve():
    pass

def vol_sheath():
    pass

def lstsqPlaneEstimation(pts):
    # pts need to be of dimension (Nx3)
    # pts need to be centered before estimation of normal!!
    ptsCentered = pts-np.mean(pts,axis=0)
    # do fit via SVD
    u, s, vh = np.linalg.svd(ptsCentered[:,0:3].T, full_matrices=True)
    #print(u)
    #print(s)
    #print(vh)
    normal = u[:,-1] 
    # the normal should point towards the world z-direction
    if normal[-1]<0:
        normal *= -1.0
    #R = np.zeros((3,3))
    #for i in range(3):
    #    R[:,i] = u[:,i] / np.linalg.norm(u[:,i])
    R = u.copy() # u is already orthonormal
    #normal = u[:,-1] / np.linalg.norm(u[:,-1])
    offset = np.mean(pts[:,0:3],axis=0)
    return (normal,offset,R)

def change_columnname_in_dictionary(dataframe, nameSeg: str):
    column_with_segment = {}
    for idx, name in enumerate(df.columns):
        column_with_segment[name] = f'{nameSeg}_{name}'
    updated_dataframe = df.rename(columns = column_with_segment, inplace = False)
    return updated_dataframe


#%% Prepare the file names 
pnMRIptn = r'C:\users\tangge\SANS\Raw_data\%s'
ptnMRI = 'Denoised_*.nii'
pnout = r'C:\users\tangge\SANS\Raw_data\Summary'
ffout = r'allFidDistances_%s'
file_extension = r'.csv'

df_fls = []
flagDemo = False
save = True
# For each cohort (Cosmonauts and Astronauts)
list_tags_cohort = []
for iterc, tag_cohort in enumerate([['Cosmo02mm', 'Cosmonaut_BIDS']]): 
    print(iterc, tag_cohort)    
    if flagDemo:
        if iterc>0:
            break
    pnMRI  = pnMRIptn%tag_cohort[1]
    if iterc == 0:
        ffout = ffout%tag_cohort[0]
    else:
        ffout = ffout + '%s' 
        ffout = ffout%tag_cohort[0]            
    fl = fs.locateFilesDf(ptnMRI, pnMRI, level=3)
    tempTagsCohort = [tag_cohort[0] for x in fl.fn]
    fl['tag_cohort'] = tempTagsCohort
    df_fls.append(fl)

fl = pd.concat(df_fls)
fl = fl.reset_index(drop=True)
fn_splits = [x.split('_') for x in fl.fn]
tagsID  = [x[0] for x in fn_splits]
tagsSes = [x[1][4:] for x in fn_splits]
fl['id'] = tagsID
fl['ses'] = tagsSes

#%% Extract the distances
fnFIDSptn = 'fids*.fcsv' # Astronauts / Cosmonauts
dfsFIDS = []
list_fid_measures = []
for idx in range(fl.shape[0]):
    #pnFIDS = r'D:\Dropbox\Projects\AstronautT1\data\results_fidsFiles_AstronautT1'
    pn, fn = os.path.split(fl.ff[idx].replace('/','\\'))
    pn = r'{}'.format(pn)    
    ffFIDS = fs.locateFiles(fnFIDSptn, os.path.join(pn,'SANS'), level=0)[0].replace('/','\\')
    if not os.path.exists(ffFIDS):
        # dummy dataframe
        dfFIDS = pd.DataFrame()
        dfsFIDS.append(dfFIDS)
        # dummy distances
        dists = {'d1_L': np.nan,
                 'd1_R': np.nan,
                 'd2_L': np.nan,
                 'd2_R': np.nan,
                 'd3_L': np.nan,
                 'd3_R': np.nan,
                 'd4_L': np.nan,
                 'd4_R': np.nan,
                 'd5_L': np.nan,
                 'd5_R': np.nan,
                 'd6_L': np.nan,
                 'd6_R': np.nan,
                 'fn_root': fl.fn_root[idx],
                 'n1_L': np.nan,
                 'n1_R': np.nan,
                 'w1_L': np.nan,
                 'w1_R': np.nan,
                 'w2_L': np.nan,
                 'w2_R': np.nan,
                 'w3_L': np.nan,
                 'w3_R': np.nan,
                 'w4_L': np.nan,
                 'w4_R': np.nan,
                 'h1_L': np.nan,
                 'h1_R': np.nan,}
        list_fid_measures.append(dists)
    else:    
        dfFIDS = readSlicerAnnotationFiducials(ffFIDS)
        dfFIDS = dfFIDS.set_index('label')
        dists = fid_measures(dfFIDS)
        dists['fn_root'] = fl.fn_root[idx][9:]
        dfsFIDS.append(dfFIDS)
        list_fid_measures.append(dists)
    print('Read fids file %d of %d'%(idx+1,fl.shape[0]))

fl['dfFIDS'] = dfsFIDS
fl['fid_measures'] = list_fid_measures
#%
dfDists = pd.DataFrame(list_fid_measures)
cols = ['fn_root', 'd1_L', 'd1_R', 'd2_L', 'd2_R', 'd3_L', 'd3_R', 'd4_L', 'd4_R', 
        'd5_L', 'd5_R', 'n1_L', 'n1_R', 
        'w1_L', 'w1_R', 'w2_L', 'w2_R', 'w3_L', 'w3_R',
        'w4_L', 'w4_R', 'h1_L', 'h1_R']

dfDists = dfDists[cols]

#%% Save the files
if save:
    ffout = (fs.osnj(pnout, ffout) + file_extension)
    print(f'Save the file to {ffout}')
    if file_extension == '.xlsx':
        dfDists.to_excel(ffout, index = False)
    elif file_extension == '.csv':
        dfDists.to_csv(ffout, index = False)

