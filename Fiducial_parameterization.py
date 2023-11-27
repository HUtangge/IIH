"""
Extract The fiducial parameters from Fiducial list
"""
#%% Configurations
project_path = r'D:\users\getang\IIH'
date = 20231127
flagDemo = False
save = True

#%%
import sys
sys.path.append(r"D:\users\getang\SANS\Slicertools")
import file_search_tool as fs
import pandas as pd
import numpy as np
import os
import re

#%% Some help functions for this part
def readSlicerAnnotationFiducials(ff):
    fids = pd.read_csv(ff,
                       comment='#',
                       header=None,
                       names=['id','x','y','z','ow','ox','oy','oz','vis','sel',
                              'lock','label','desc','associatedNodeID', 'unknown1', 'unknown2'],
                       engine='python')
    return fids

def df_dist(df,pt1,pt2):
    p1 = df.loc[pt1,['x','y','z']].values.astype(float)
    p2 = df.loc[pt2,['x','y','z']].values.astype(float)
    return np.linalg.norm(p1-p2)

def fid_measures_T1(df, withEyeOrbDist=True):
    #df = df.set_index('label')
    d = dict()
    for side in ['L','R']: 
        d['d1_%s'%side] = df_dist(df,'center_%s_lens'%side,'center_%s_eyeball'%side)
        d['d2_%s'%side] = df_dist(df,'center_%s_eyeball'%side,'nerve_tip_%s'%side)
        d['d3_%s'%side] = df_dist(df,'center_%s_lens'%side,'eyeball_back_%s'%side)
        d['w1_%s'%side] = df_dist(df,'eyeball_midline_%s_lat'%side,'eyeball_midline_%s_med'%side)
        d['w2_%s'%side] = df_dist(df,'nerve_baseline_muscle_%s_lat'%side,'nerve_baseline_muscle_%s_med'%side)
        d['w3_%s'%side] = df_dist(df,'nerve_baseline_bone_%s_lat'%side,'nerve_baseline_bone_%s_med'%side)
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
            ptsEyeCtr  = df.loc['center_%s_eyeball'%side,['x','y','z']].values.astype(float)
            disteye = np.dot(normal,ptsEyeCtr-ptsOrbMean)
            d['d4_%s'%side] = disteye
            # Lens center to the plane
            ptsLensCtr  = df.loc['center_%s_lens'%side,['x','y','z']].values.astype(float)
            distlens = np.dot(normal,ptsLensCtr-ptsOrbMean)
            d['d5_%s'%side] = distlens
    return d

def fid_measures_T2(df, withEyeOrbDist=True):
    #df = df.set_index('label')
    d = dict()
    for side in ['L','R']: 
        d['d1_%s'%side] = df_dist(df,'center_%s_lens'%side,'center_%s_eyeball'%side)
        d['d2_%s'%side] = df_dist(df,'center_%s_eyeball'%side,'nerve_tip_%s'%side)
        d['d3_%s'%side] = df_dist(df,'center_%s_lens'%side,'eyeball_back_%s'%side)
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
            ptsEyeCtr  = df.loc['center_%s_eyeball'%side,['x','y','z']].values.astype(float)
            disteye = np.dot(normal,ptsEyeCtr-ptsOrbMean)
            d['d4_%s'%side] = disteye
            # Lens center to the plane
            ptsLensCtr  = df.loc['center_%s_lens'%side,['x','y','z']].values.astype(float)
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
pnMRI = r'%s\data\Rawdata'%project_path
fnMRIptn = r'Denoised_*_%sw.nii'
pnout = r'%s\data\Rawdata\Summary'%project_path
ffout = r'allFidDistances_%s'%date
file_extension = '.csv'
fnFIDSptn = r'fids*%sw.fcsv'

df_fls = []
# For each modality (Cosmonauts and Astronauts)
list_modality = []
for iterc, modality in enumerate([['IIH02mm', 'T1'], ['IIH02mm', 'T2']]): 
    print(iterc, modality)    
    if flagDemo & iterc>0:
        break
    fnMRI = fnMRIptn%(modality[1])
    fl = fs.locateFilesDf(fnMRI, pnMRI, level=1)

    tempTagsmodality = [modality[1] for x in fl.fn]
    fl['modality'] = tempTagsmodality
    df_fls.append(fl)

fl = pd.concat(df_fls)
fl = fl.reset_index(drop=True)
fn_splits = [x.split('_') for x in fl.fn]
tagsID  = [f'{x[1]}_{x[2]}' for x in fn_splits]
tagsSub  = [x[1] for x in fn_splits]
tagsSes = [x[2] for x in fn_splits]
fl['id'] = tagsID
fl['sub'] = tagsSub
fl['ses'] = tagsSes

#%% Extract the distances
list_fid_measures = []
for idx in range(fl.shape[0]):
    #pnFIDS = r'D:\Dropbox\Projects\AstronautT1\data\results_fidsFiles_AstronautT1'
    pn, fn = os.path.split(fl.ff[idx].replace('/','\\'))
    pn = r'{}'.format(pn)       
    fn_splits = fn.split('_')

    if bool(re.search('T1', fn)):
        modality='T1'
    elif bool(re.search('T2', fn)):
        modality='T2'
    ffFIDS = fs.locateFiles(fnFIDSptn%modality, os.path.join(pn,'Metrics'), level=0)[0].replace('/','\\')
    if not os.path.exists(ffFIDS):
        # dummy dataframe
        dfFIDS = pd.DataFrame()
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
    else:    
        dfFIDS = readSlicerAnnotationFiducials(ffFIDS)
        dfFIDS = dfFIDS.set_index('label')
        if modality=='T1':
            dists = fid_measures_T1(dfFIDS)
        elif modality=='T2':
            dists = fid_measures_T2(dfFIDS)
    dists['fn_root'] = fl.fn_root[idx][9:]
    dists.update({'id':f'{fn_splits[1]}_{fn_splits[2]}'})
    dists.update({'sub':fn_splits[1]})
    dists.update({'ses':fn_splits[2]})
    list_fid_measures.append(dists)        
    print('Read fids file %d of %d'%(idx+1,fl.shape[0]))

dfDists = pd.DataFrame.from_records(list_fid_measures)

if save:
    ffout = (fs.osnj(pnout, ffout) + file_extension)
    print(f'Save the file to {ffout}')
    if file_extension == '.xlsx':
        dfDists.to_excel(ffout, index = False)
    elif file_extension == '.csv':
        dfDists.to_csv(ffout, index = False)

