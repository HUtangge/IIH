"""
Ge Tang

Conda enviroment for running this script:
    conda activate astronaut_sans

Explantion for the plot
The x y represents the view from the front of the eyeball, 
x represents the nasal and temporal direction,
    nasal is negative and temporal is positive
y represetns the superior and inferior
    superior is positive and inferior is negative

"""

#%% Configurations
project_path = r'/Users/getang/Documents/EarthResearch/IIH'
save = False
plotting = True
projection_map = 'Polar'
degree_to_center = 90
side_of_eye = ['L', 'R'] # choices are 'R', 'L', and ['L', 'R']
modality = 'T1' # For IIH cohort, the modality is T1 and T2
contour_level = 200
# Parameters for the size of degree on the plot
# Define the radii for the three circles
radii = [(3/270)*degree_to_center, (2/270)*degree_to_center, (1/270)*degree_to_center]
x_offset = 0.02  # Adjust this value to control the horizontal space
y_offset = 0.02 
fontsize = 18

#%% create a sphere
import sys
import os
sys.path.append(os.path.join(project_path, 'src'))

import file_search_tool as fs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import random
import pickle
import vtk

# Functions
def open_pickle(filename:str):
    infile = open(filename,'rb')
    temp_data = pickle.load(infile)
    infile.close()
    return temp_data

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

def plot_2d_contour(grid_x, grid_y, grid_z, plotname = None, levels=1000, cmap='coolwarm', projection_map = 'Polar'):
    if grid_z.ndim == 2:
        # Create a plot
        fig, ax = plt.subplots()
        norm = plt.Normalize(vmin, vmax)
        ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, norm = norm)
        ax.plot(0, 0, marker=".", markersize=10, markeredgecolor="red")

        # if testing:
        #     point_list = np.array([100, 101, 102, 103, 104, 105]) * 20 + 10
        #     print(point_list)
        #     plt.scatter(data[point_list,0], data[point_list,1], color='black', marker='.', label='Points')

        if projection_map == 'Polar':
            # Plot the circles on ax1
            for radius in radii:
                circle = plt.Circle((0, 0), radius=radius, fill=False, color='black', lw=1)
                plt.gca().add_artist(circle)
                plt.text(radius - x_offset, 0, f'{int(90*radius)}°', ha='right', va='center', fontsize=fontsize)

        # Create a ScalarMappable with the desired normalization
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        # Add a colorbar with symmetric range
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Colorbar')

        if plotname == None:
            plt.title('2D Contour Plot of back of eyeball')
        else:
            plt.title(plotname)
        
        if 'L_' in plotname:
            plt.xlabel('Temporal to Nasal')
        else:
            plt.xlabel('Nasal to Temporal')

        plt.ylabel('Inferior to Superior')
        plt.savefig(f'{pnpolarprojection}/{plotname}.pdf', format='pdf')
        plt.show()

    elif grid_z.ndim == 3:
        # Create a figure
        fig = plt.figure(figsize=(14, 8))

        # Create a GridSpec with equal-sized cells
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

        # Create subplots with equal aspect ratio
        ax1 = plt.subplot(gs[0], aspect='equal')
        ax2 = plt.subplot(gs[1], aspect='equal')

        # Normalize data based on the global minimum and maximum values across both datasets
        norm = plt.Normalize(vmin, vmax)

        # Plot contour for dataset1
        ax1.contourf(grid_x, grid_y, grid_z[:,:,0], levels=levels, cmap=cmap, norm=norm)
        ax1.plot(0, 0, marker=".", markersize=8, markeredgecolor="red")
        if plotname == None:
            ax1.set_title('2D Contour Plot of back of eyeball')
        else:
            ax1.set_title(plotname[0][0])
        ax1.set_xlabel('Temporal to Nasal')
        ax1.set_ylabel('Inferior to Superior')

        # Plot contour for dataset2
        ax2.contourf(grid_x, grid_y, grid_z[:,:,1], levels=levels, cmap=cmap, norm=norm)
        ax2.plot(0, 0, marker=".", markersize=8, markeredgecolor="red")
        if plotname == None:
            ax2.set_title('2D Contour Plot of back of eyeball')
        else:
            ax2.set_title(plotname[1][0])
        ax2.set_xlabel('Nasal to Temporal')
        # ax2.set_ylabel('Inferior to Superior')

        if projection_map == 'Polar':
            # Plot the circles on ax1
            for radius in radii:
                circle = Circle((0, 0), radius=radius, fill=False, color='black', lw=1)
                ax1.add_patch(circle)
                ax1.text(radius - x_offset, 0, f'{int(90*radius)}°', ha='right', va='center', fontsize=fontsize)
            # Plot the circles on ax2
            for radius in radii:
                circle = Circle((0, 0), radius=radius, fill=False, color='black', lw=1)
                ax2.add_patch(circle)
                ax2.text(radius - x_offset, 0, f'{int(90*radius)}°', ha='right', va='center', fontsize=fontsize)
        
        # Create a ScalarMappable with the desired normalization
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cax = plt.subplot(gs[2])
        # ticks 
        step = 0.05
        ticks = np.arange(np.floor(vmin / step) * step, np.ceil(vmax / step) * step + step, step)

        cbar = fig.colorbar(sm, cax=cax, orientation='vertical', ticks = ticks)
        cbar.set_label('Colorbar')
        plt.savefig(f'{pnpolarprojection}/{plotname[0][0][2:]}.pdf', format='pdf')
        plt.show()

#%% Names of the files
pnpolarprojection = fs.osnj(project_path, 'data')
fnpolarprojection = f'{projection_map}_projection_*.vtp'
fnsummary = f'{projection_map}_projection_eyeball_{degree_to_center}_summary.npy'
polargrid = f'{projection_map}_grid_eyeball_{degree_to_center}_summary.npy'
plotname = f'{projection_map} projection of the back of the eyeball'
info_filename = f'Polar_projection_info.csv'

#%% Load data and concatenate all the data
if save:
    pnpolarprojection = fs.osnj(project_path, 'data')
    fl = fs.locateFiles(fnpolarprojection, fs.osnj(pnpolarprojection, f'eyeball_polar_projection'), level=0)
    for idx,file in enumerate(fl):
        print(file)
        if projection_map == 'Orthographic':
            # To distinguish the projection map, save them into different data format
            print('Loading Orthographic projected data')
            data = open_pickle(file)
        elif projection_map == 'Polar':
            print('Loading Polar coordinate transformed data')
            data = vtk_points_to_numpy_array(load_vtk_points(file))

        # Stack all the data into a 3D array, while keeping the grid also saved
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        # Create a grid for interpolation
        X, Y = np.mgrid[min(x):max(x):1000j, min(y):max(y):1000j]
        # Interpolate z values on the grid
        Z = griddata((x, y), z, (X, Y), method='linear')
        Z = np.expand_dims(Z, axis=-1)
        if idx == 0:
            PolarprojectionMetrics = Z
            X = np.expand_dims(X, axis=-1)
            Y = np.expand_dims(Y, axis=-1)
            Grids = np.concatenate((X, Y), axis=2)
        else:
            PolarprojectionMetrics = np.concatenate((PolarprojectionMetrics, Z), axis=2)
    # Save the files
    print(f'Saving file to {os.path.join(pnpolarprojection, fnsummary)}')
    np.save(os.path.join(pnpolarprojection, fnsummary), PolarprojectionMetrics)
    np.save(os.path.join(pnpolarprojection, polargrid), Grids)

#%% Pickup the specific rows for the mean image
if plotting:
    # Configuration for the plotting
    df_info = pd.read_csv(fs.osnj(project_path, 'data', info_filename))
    if not (('R' in side_of_eye) and ('L' in side_of_eye)):
        # Load the 3D data and preparing plotting
        PolarprojectionMetrics = np.load(os.path.join(pnpolarprojection, fnsummary))
        Grids = np.load(os.path.join(pnpolarprojection, polargrid))
        X = Grids[:,:,0]
        Y = Grids[:,:,1]
        condition = input("Please enter which condition for plotting: e.g., control, variance, cosmonaut_preflight, cosmonaut_postflight1, cosmonaut_followup, preflight-postflight1, preflight-followup ... ")
        if condition == 'control':
            cmap = cm.hot.reversed()
            file_idx = df_info[(df_info['group'] == 'control') & (df_info['side_of_eyeball'] == side_of_eye) & (df_info['modality'] == modality)].index.values
            file_idx = np.squeeze(file_idx)
            PolarprojectionMetrics = PolarprojectionMetrics[:,:,file_idx]
            Z = np.mean(PolarprojectionMetrics, axis=2)    
            vmin = np.nanmin(Z)
            vmax = np.nanmax(Z)
        elif condition == 'variance':
            cmap = cm.hot.reversed()
            file_idx = df_info[(df_info['group'] == 'control') & (df_info['side_of_eyeball'] == side_of_eye) & (df_info['modality'] == modality)].index.values
            file_idx = np.squeeze(file_idx)
            PolarprojectionMetrics = PolarprojectionMetrics[:,:,file_idx]
            Z = np.var(PolarprojectionMetrics, axis=2)
            vmin = np.nanmin(Z)
            vmax = np.nanmax(Z)

        plotname = f'{side_of_eye}_back_of_eyeball_{projection_map}_projection_for_{condition}'    
        plot_2d_contour(X, Y, Z, plotname = plotname, cmap = cmap, levels = contour_level)
    
    elif ('R' in side_of_eye) and ('L' in side_of_eye):
        print('Plotting both eyes')
        # Load the 3D data and preparing plotting
        condition = input("Please enter which condition for plotting: e.g., control, variance, cosmonaut_preflight, cosmonaut_postflight1, cosmonaut_followup, preflight-postflight1, preflight-followup ... ")
        fnsummary = f'{projection_map}_projection_eyeball_{degree_to_center}_summary.npy'
        polargrid = f'{projection_map}_grid_eyeball_{degree_to_center}_summary.npy'

        # Loading the Polar Metrics for both eye and store them into a 4D array
        PolarprojectionMetrics = np.expand_dims(np.load(os.path.join(pnpolarprojection, fnsummary)), axis = -1)
        for idx, eye in enumerate(side_of_eye):
            if condition == 'variance':
                print(f'selecting Controls')
                file_idx = df_info[(df_info['group'] == 'control') & (df_info['side_of_eyeball'] == eye) & (df_info['modality'] == modality)].index.values
            else:
                file_idx = df_info[(df_info['group'] == condition) & (df_info['side_of_eyeball'] == eye) & (df_info['modality'] == modality)].index.values
            
            file_idx = np.squeeze(file_idx)
            plotname = list([f'{eye}_back_of_eyeball_{projection_map}_projection_for_{condition}'])
            if idx == 0:
                PolarprojectionMetrics4D = PolarprojectionMetrics[:,:,file_idx,:]
                plotnameall = list([plotname])
            else:
                PolarprojectionMetrics4D = np.concatenate((PolarprojectionMetrics4D, PolarprojectionMetrics[:,:,file_idx,:]), axis=3)
                plotnameall.append(plotname)

        # Get the Grids
        Grids = np.load(os.path.join(pnpolarprojection, polargrid))
        X = Grids[:,:,0]
        Y = Grids[:,:,1]

        if condition == 'control':
            cmap = cm.hot.reversed()
            # cmap = cm.coolwarm
            Z = np.mean(PolarprojectionMetrics4D, axis=2)
            vmin = np.nanmin(Z)
            vmax = np.nanmax(Z)
        elif condition == 'variance':
            cmap = cm.hot.reversed()
            Z = np.var(PolarprojectionMetrics4D, axis=2)
            vmin = np.nanmin(Z)
            vmax = np.nanmax(Z)

        plot_2d_contour(X, Y, Z, plotname = plotnameall, cmap = cmap, levels = contour_level)

if __name__ == 'main': 
    print('Done')
    # python -c "import matplotlib; print(matplotlib.__version__)"
    # python -c "import pandas; print(pandas.__version__)"



# %%
