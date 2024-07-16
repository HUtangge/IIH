#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:46 2023

@author: getang

Run in Slicer 4.11.20200930 at work station computer

"""
#%% Configurations
# project_path = r'/Users/getang/Documents/SpaceResearch/spaceflight_associated_neuroocular_syndrome/SANS'
project_path = r'D:\users\getang\IIH'
Orthogonal_projection = False
scaling = 50 # Scalling for finding the intersection of lines on the eyeball plane
degree_to_center = 90
sampling_points = int(1e4)
save = True
testing = False

#%%
import sys
import os
# sys.path.append(os.path.join(project_path, 'src'))
sys.path.append(os.path.join(project_path, 'Slicertools'))

import file_search_tool as fs
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import math
from math import sqrt, cos, sin, acos
import random
# from importlib import reload
# reload(su)

# %% functions
def point_to_vector(point):
    return np.array(point)

def vector_to_point(vector):
    return tuple(vector)

def intersection_line(plane1, plane2):
    # Calculate the direction vector of the intersection line
    normal1 = np.array(plane1.GetNormal())
    normal2 = np.array(plane2.GetNormal())
    direction = np.cross(normal1, normal2)
    if np.linalg.norm(direction) == 0:
        return None, None  # Planes are parallel and don't intersect
    # Find a point on the intersection line
    origin1 = np.array(plane1.GetOrigin())
    origin2 = np.array(plane2.GetOrigin())
    A = np.vstack((normal1, normal2, direction))
    b = np.array([np.dot(normal1, origin1), np.dot(normal2, origin2), 0])
    try:
        point_on_line = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None  # Planes coincide or the system has no unique solution
    return point_on_line, direction

def unit_axis_angle(a, b):
    an = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    bn = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2])
    ax, ay, az = a[0]/an, a[1]/an, a[2]/an
    bx, by, bz = b[0]/bn, b[1]/bn, b[2]/bn
    nx, ny, nz = ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx
    nn = sqrt(nx*nx + ny*ny + nz*nz)
    axis = np.array([nx/nn, ny/nn, nz/nn])
    angle = acos(ax*bx + ay*by + az*bz)
    return axis, angle

def rotation_matrix(axis, angle):
    ax, ay, az = axis[0], axis[1], axis[2]
    s = sin(angle)
    c = cos(angle)
    u = 1 - c
    return (np.array([[ax*ax*u + c, ax*ay*u - az*s, ax*az*u + ay*s],
             [ay*ax*u + az*s, ay*ay*u + c, ay*az*u - ax*s],
             [az*ax*u - ay*s, az*ay*u + ax*s, az*az*u + c]]))

def get_intersection_points(pd, start_point, end_point):
    start_point = start_point.reshape(3,1)
    end_point = end_point.reshape(3,1)
    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(pd)
    obb_tree.BuildLocator()
    # Perform intersection computation
    intersection_points = vtk.vtkPoints()
    obb_tree.IntersectWithLine(start_point, end_point, intersection_points, None)
    intersection_points_array = vtk_to_numpy(intersection_points.GetData())
    intersection_points_array = intersection_points_array.reshape(-1, 3)
    return intersection_points_array

def point_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def scale_polar_angle(polar_angle, max_polar_angle):
    return polar_angle / max_polar_angle

def spherical_to_cartesian(radial, azimuthal_array, polar_array, max_polar_angle, real_radius, clockwise=True):
    """
    Convert spherical coordinates to cartesian coordinates.
    :param radial: the normalization for the azimzuthal and polar angles
    :param azimuthal_array: array of azimuthal angles (in degeres) n by 1 matrix
    :param polar_array: array of polar angles (in degrees) n by 1 matrix
    :param max_polar_angle: maximum polar angle (in degrees) a number
    :param real_radius: real radius of the sphere, n by 1 matrix
    :return: vtkPoints object
    """
    vtk_points = vtk.vtkPoints()
    for azimuthal, polar, radius in zip(azimuthal_array, polar_array, real_radius):
        r = radial * scale_polar_angle(polar, max_polar_angle)
        # The np.cos only expects radians
        if not clockwise:
            azimuthal = - azimuthal
        x = r * np.cos(np.radians(azimuthal))
        y = - r * np.sin(np.radians(azimuthal)) # Flip the coordinate to make it consistent with the orthogonal projection map 
        z = radius
        vtk_points.InsertNextPoint(x, y, z)
    return vtk_points

def radius(x, y):
    return np.sqrt(x ** 2 + y ** 2)

def save_vtk_points(vtk_points, filename):
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(poly_data)
    writer.Write()

def vtk_points_to_numpy_array(vtk_points):
    num_points = vtk_points.GetNumberOfPoints()
    data = np.zeros((num_points, 3))
    for i in range(num_points):
        point = vtk_points.GetPoint(i)
        data[i, 0] = point[0]
        data[i, 1] = point[1]
        data[i, 2] = point[2]
    return data

def dot_product(a, b):
    return np.dot(a, b)

def magnitude(a):
    return np.linalg.norm(a)

def cross_product(a, b):
    return np.cross(a, b)

def angles_between_vectors(matrix_a, vector_b):
    angles = np.zeros(len(matrix_a))
    for i in range(matrix_a.shape[0]):
        a = matrix_a[i]
        b = vector_b[0]
        dot_prod = dot_product(a, b)
        mag_a = magnitude(a)
        mag_b = magnitude(b)
        cos_theta = dot_prod / (mag_a * mag_b)
        theta_rad = math.acos(cos_theta)
        theta_deg = math.degrees(theta_rad)
        cross_prod = cross_product(a, b)
        if cross_prod[2] < 0:
            theta_deg = 360 - theta_deg
        angles[i] = theta_deg
    angles = angles.reshape(-1, 1)
    return angles

#%%
# loop over the subjects to Get the individual eyeball mapping
fnSEGSptn = r'Segs_*_%sw.seg.nrrd'
fnFIDSptn = r'fids_*_%sw.fcsv'
pnMRI = os.path.join(project_path, 'data', 'Rawdata')
fnMRIptn = r'Denoised_*_%sw.nii'

for iterc, modality in enumerate([['IIH02mm', 'T1'], ['IIH02mm', 'T2']]): 
    print(iterc, modality)    
    fnSEGS = fnSEGSptn%(modality[1])
    fnFIDS = fnFIDSptn%(modality[1])
    fnMRI = fnMRIptn%(modality[1])

    # Attention: Locate file use fnmatch, which is different from regular expression
    fl = fs.locateFiles(fnMRI, pnMRI, level=1)
    
    # for each file
    for idx, ff in enumerate(fl):
        print(ff)
        pn, fn = os.path.split(ff)
        pn = r'{}'.format(pn)
        fnroot, fnext = fs.splitext(fn) 
        # Load Fiducials
        fffids = fs.locateFiles(fnFIDS, fs.osnj(pn, 'Metrics'), level=0)
        if len(fffids) == 1:
            nFIDS = loadMarkupsFiducialList(fffids[0], returnNode=True)
            nFIDS.SetName('fids_IIH02mm')
        else:
            message = f'More than one fiducial file found! {fffids}.'
            warnings.warn(message)
        # Load Segmentations
        ffSeg = fs.locateFiles(fnSEGS, fs.osnj(pn, 'Metrics'), level=0)
        if len(ffSeg) == 1:
            success,nSEGS = loadSegmentation(ffSeg[0], returnNode=True)
            nSEGS.SetName('Seg_IIH02mm')
        else:
            message = f'More than one segmentation file found! {ffSeg}.'
            warnings.warn(message)
        # Set the parameters for getting the projection mapping
        # for name in ['R_eyeball', 'L_eyeball']:
        for name in ['R_eyeball', 'L_eyeball']:
            print(f'Calculate the {name} projection mapping')
            n = su.getNode('Seg_IIH02mm')
            s = n.GetSegmentation()
            ss = s.GetSegment(s.GetSegmentIdBySegmentName(name))
            pd = ss.GetRepresentation('Closed surface')
            nervetip = su.arrayFromFiducialList('fids_IIH02mm', list(['nerve_tip_' + name.split('_')[0]]))
            # Get the center
            center = np.array(pd.GetCenter())
            center = center.reshape(1,3)
            # Get all the points on the surface
            points = np.array(pd.GetPoints().GetData())
            moving_vector = points - center
            # This is the z axis in the polar coordinate system
            axis_vector = nervetip - center
            polar_angles = su.getAngle(moving_vector, axis_vector, radians = False)
            if Orthogonal_projection:
                projectionname = 'Orthognal_projection' + '_' + name + '_' + str(degree_to_center) + '_' + fnroot
                # Add the polar_angles to the info
                points_info = np.append(points, polar_angles, axis=1)
                # Add the length to the info
                points_info = np.append(points_info, su.getLength(moving_vector), axis=1)
                # For tesitng the code, we can see a hole in the middle of the eyeball
                point_list = points_info[:,3] < degree_to_center
                points_info = points_info[point_list]
                # Create the plane for orthogonal projection
                plane = su.getPlane(axis_vector, center)
                eye_plane_xy = su.getOrCreateMarkupsPlaneNode('eye_plane', center, axis_vector.reshape(3,1))
                planecenter = center.reshape(3,1)
                planenormal = (axis_vector / np.linalg.norm(axis_vector).T).reshape(3,1)
                pointsonplane = np.zeros((len(points_info),3))
                # plane is the vtkPlane object
                for i in range(len(points_info)):
                    point = points_info[i,0:3]
                    planeprojection = np.zeros((3,1))
                    plane.ProjectPoint(point.reshape(3,1), planecenter, planenormal, planeprojection)
                    pointsonplane[i,] = planeprojection.reshape(1,3)
                pointsonplane_info = np.append(pointsonplane, su.getLength(moving_vector[point_list]), axis=1)
                # Get the coordinate on the projection plane.
                ref_plane_xy = su.getOrCreateMarkupsPlaneNode('reference_plane', np.array([0,0,0]).reshape(1,3), np.array([0,0,1]).reshape(3,1))
                point_on_line, direction_x = intersection_line(ref_plane_xy, eye_plane_xy)
                if name.split('_')[0] == 'L':
                    # For flip the right and left side
                    direction_x = direction_x
                axis1_line = get_intersection_points(pd, direction_x * scaling + center, - direction_x * scaling + center)
                orthogonal_vector = np.cross(direction_x, eye_plane_xy.GetNormal())
                orthogonaline = np.array([center + orthogonal_vector * 50, center - orthogonal_vector * 50]).reshape(2,3)
                if testing:
                    su.getOrCreateMarkupsLineNode('orthogonal_line', orthogonaline)
                axis2_line = get_intersection_points(pd, center + orthogonal_vector * 50, center - orthogonal_vector * 50)
                axis_points = np.array([axis1_line, axis2_line]).reshape(4,3)
                # Get the rotation matrix (New way)
                # Coordinate system A
                center_A = np.array([0, 0, 0])
                x_axis_A = np.array([1, 0, 0])
                y_axis_A = np.array([0, 1, 0])
                z_axis_A = np.array([0, 0, 1])
                # Coordinate system B
                center_B = center
                x_axis_B = direction_x # Nasel and temporal
                y_axis_B = -orthogonal_vector # superior and inferior
                z_axis_B = eye_plane_xy.GetNormal()
                # Normalize the basis vectors
                x_axis_B_normalized = normalize(x_axis_B)
                y_axis_B_normalized = normalize(y_axis_B)
                z_axis_B_normalized = normalize(z_axis_B)
                R_A_to_B = np.column_stack((x_axis_B_normalized, y_axis_B_normalized, z_axis_B_normalized))
                # print(R_A_to_B)
                # Compute the rotation matrix from B to A
                R_B_to_A = R_A_to_B.T
                pointsonplane_offset = pointsonplane[:,:3] - center
                rotated = np.dot(R_B_to_A, pointsonplane_offset.T).T
                rotated_xyaxis = np.dot(R_B_to_A, (axis_points-center).T).T
                rotated_info = np.append(rotated[:,:2], su.getLength(moving_vector[point_list]), axis=1)
                if save:
                    su.save_pickle(rotated_info, fs.osnj(pn, 'Metrics', projectionname + '.pickle'))
            else:
                projectionname = 'Polar_projection' + '_' + name + '_' + str(degree_to_center) + '_' + fnroot
                # Get the coordinate on the projection plane.
                plane = su.getPlane(axis_vector, center)
                eye_plane_xy = su.getPlane(axis_vector, center)
                ref_plane_xy = su.getPlane(np.array([0,0,1]).reshape(3,1), np.array([0,0,0]).reshape(1,3))
                # eye_plane_xy = su.getOrCreateMarkupsPlaneNode('eye_plane', center, axis_vector.reshape(3,1))
                # ref_plane_xy = su.getOrCreateMarkupsPlaneNode('reference_plane_xy', np.array([0,0,0]).reshape(1,3), np.array([0,0,1]).reshape(3,1))
                # In the right eye exmaple, the direction_x is pointing to the temporal direction
                point_on_line, direction_x = intersection_line(ref_plane_xy, eye_plane_xy)
                # correct the sign of the projections, temporal is always positive
                if name.split('_')[0] == 'L':
                    # For flip the right and left side
                    # direction_x = - direction_x
                    direction_x = direction_x
                # Get the projected points on the plane to calculate the azimuthal angle
                planecenter = center.reshape(3,1)
                planenormal = (axis_vector / np.linalg.norm(axis_vector).T).reshape(3,1)
                pointsonplane = np.zeros((len(points),3))
                for i in range(len(points)):
                    point = points[i,:]
                    planeprojection = np.zeros((3,1))
                    plane.ProjectPoint(point.reshape(3,1), planecenter, planenormal, planeprojection)
                    pointsonplane[i,] = planeprojection.reshape(1,3)
                projected_moving_vector = pointsonplane - center
                # Get the azimuthal angle, the temporal direction as the positive direction of x axis
                azimuthal_angles = angles_between_vectors(projected_moving_vector, direction_x.reshape(1,3))
                # Create the point matrix with the polar coordinate system infomation
                points_info = np.append(points, azimuthal_angles, axis=1)
                points_info = np.append(points_info, polar_angles, axis=1)
                points_info = np.append(points_info, su.getLength(moving_vector), axis=1)
                # Select the points that are in the south hemisphere based on the polar angle
                south_point_list = points_info[:,4] < degree_to_center
                points_info = points_info[south_point_list]
                if name.split('_')[0] == 'L':
                    # cartesian_coord = spherical_to_cartesian(1, points_info[:,3], points_info[:,4], 90, points_info[:,5], clockwise=False)
                    cartesian_coord = spherical_to_cartesian(1, points_info[:,3], points_info[:,4], 90, points_info[:,5])
                else:
                    cartesian_coord = spherical_to_cartesian(1, points_info[:,3], points_info[:,4], 90, points_info[:,5])
                if save:
                    save_vtk_points(cartesian_coord, fs.osnj(pn, 'Metrics', projectionname + '.vtp'))
        
        if not testing:
            print('Close the Scene')
            su.closeScene()       

#%%
if __name__ == '__main__':
    # Add the fiducials to the scene in order to visualize the result
    su.fiducialListFromArray('points_on_eyeball', points_info[points_info[:,3] <1,:3], listFidNames=None)
    
    # Create a reference plane
    plane = su.getPlane(axis_vector, center)
    projectplane = su.getOrCreateMarkupsPlaneNode('reference_plane', np.array([0,0,0]).reshape(1,3), np.array([0,0,1]).reshape(3,1))

    ref_plane_xz = su.getOrCreateMarkupsPlaneNode('reference_plane', np.array([0,0,0]).reshape(1,3), np.array([0,1,0]).reshape(3,1))

    random_integers = [random.randint(1, rotated.shape[0]) for _ in range(100)]

    su.fiducialListFromArray('pointsonplane', pointsonplane[random_integers,:])
    su.fiducialListFromArray('rotated', rotated[random_integers,:])

    d1_pointsonplane = point_distance(pointsonplane[3000,:], pointsonplane[7000,:])
    d2_rotated = point_distance(rotated[3000,:], rotated[7000,:])
    d1_pointsonplane == d2_rotated

    su.getOrCreateMarkupsLineNode('ax1', axis_points[2:,])
    su.getOrCreateMarkupsLineNode('ax2', rotated_xyaxis[2:,])