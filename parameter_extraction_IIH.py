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
Metric4Merge.set_index('id', inplace = True)
Eyeball_dist4Merge = eyeball_dist.copy()
Eyeball_dist4Merge.set_index('id', inplace = True)

# Merge the Metrics
MergedMetrics = pd.concat([Metric4Merge, Eyeball_dist4Merge], axis = 1)

# Save the Metrics
if save:   
    print(f'Saving the Metrics to the {project_path_pc}')
    MergedMetrics.to_csv(os.path.join(project_path_pc, f'{Metrics_name}.csv'), index = False)
    MergedMetrics.to_excel(os.path.join(project_path_pc, f'{Metrics_name}.xlsx'), index = False)


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

    