# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:20:56 2021

@author: getang
"""

import sys
sys.path.append(r"D:\users\getang\SANS\Slicertools")
import file_search_tool as fs
import os
import re
import time
import csv
import logging

#%% Functions for this script
def save_listdict_to_csv(data:list, namelist:list, filename:str):
    keys = set(key for dct in data for key in dct.keys())
    keys.add('filename')
    keys.add('modality')
    # Open a CSV file for writing
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        # Create a DictWriter object with the determined keys
        writer = csv.DictWriter(file, fieldnames=keys)
        # Write the header (field names)
        writer.writeheader()
        # Write the dictionary data
        for row, item in zip(data, namelist):
            print(len(row.keys()))
            item = {'filename':item}
            if len(row) < 150:
                modality = {'modality':'T1'}
            else:
                modality = {'modality':'T2'}
            row.update(item)
            row.update(modality)
            writer.writerow(row)

"""
Tangge: This is the code for 09062022
Rerun to get the volumn of the eyeball and the individualized center of eyeball and lens
"""
#%% RUN IN SLICER 
# Configurations
project_path = r'D:\users\getang\IIH'
flagDemo = True
flagSaveVisualization = True # Save the image

# Naming the destination for the transformers and the segmentations
pnATL = r'%s\template\High_resolution_template'%project_path
fnATLptn = 'T_%s%s.nii.gz'
pnFIDS = r'%s\data\Segmentations'%project_path
fnSEGSptn = r'Segs_%s%s.seg.nrrd' # Way to use: name = fnSEGSptn %(a,b)
fnFIDSptn = r'fids_%s%s.fcsv'
pnMRI = r'%s\data\Rawdata'%project_path
ptnMRI = r'Denoised_*_%sw.nii'
summary_path = r'%s\data\Rawdata\Summary\volumetrics_forall_IIH_25112023.csv'%project_path
df_fls = []
VolumeMetrics_forall = []

#%% The main loop
# for each cohort
for iterc, modality in enumerate([['IIH02mm', 'T1'], ['IIH02mm', 'T2']]): 
    print(iterc, modality)    
    fnATL  = fnATLptn%(modality[0],modality[1])
    fnSEGS = fnSEGSptn%(modality[0],modality[1])
    fnFIDS = fnFIDSptn%(modality[0],modality[1])
    ptnMRI = r'Denoised_*_%sw.nii'%(modality[1])

    # Attention: Locate file use fnmatch, which is different from regular expression
    fl = fs.locateFiles(ptnMRI, pnMRI, level=1)
    df_fls.append(fl)
    
    # for each file
    for idx, ff in enumerate(fl):
        ff = ff.replace('/','\\')
        logging.info(f'Processing {idx} {ff}')
        print(f'Processing {idx} {ff}')
        su.setLayout(3)

        if flagDemo & idx == 1:
            print('End of processing, breaking the loop')
            time.sleep(3)
            su.closeScene()       
            break  
            
        # Load the model and landmarks
        success,nATL = loadVolume(os.path.join(pnATL,fnATL),returnNode=True)
        nFIDS = loadMarkupsFiducialList(su.osnj(pnFIDS,fnFIDS),returnNode=True)
        success,nSEGS = loadSegmentation(os.path.join(pnFIDS,fnSEGS), returnNode=True)        
        
        # load T1 volume
        success,nT1 = loadVolume(ff,returnNode=True)        
        # extract fn root: sub-cosmonaut01_ses-postflight_T1w_n4
        pn, fn = os.path.split(ff)
        pn = r'{}'.format(pn)
        fnroot, fnext = fs.splitext(fn)        
        which_sub = re.search('sub-(.+?)_ses', fnroot).group(1)
        fnAFF = 'trf-Temp_to_%s_AFF-%s.mat'%(which_sub, fnroot)
        fnDEF = 'trf-Temp_to_%s_DEF-%s.nii.gz'%(which_sub, fnroot)        
        
        # Load the transforms
        success, nTrfATLtoT1_AFF = slicer.util.loadTransform(os.path.join(pn,fnAFF), returnNode=True)
        success, nTrfATLtoT1_DEF = slicer.util.loadTransform(os.path.join(pn,fnDEF), returnNode=True)        
        nTrfATLtoT1_AFF.SetAndObserveTransformNodeID(nTrfATLtoT1_DEF.GetID())
        nATL.SetAndObserveTransformNodeID(nTrfATLtoT1_AFF.GetID())
        nFIDS.SetAndObserveTransformNodeID(nTrfATLtoT1_AFF.GetID())        
        nSEGS.SetAndObserveTransformNodeID(nTrfATLtoT1_AFF.GetID())                
        
        # Save the transformed segmentation and fiducials
        nFIDS.HardenTransform() # IMPORTANT!!
        nSEGS.HardenTransform() # IMPORTANT!!            
        
        # Save the Metrics calculated in slicer
        volume_stats = su.segmentationGetsegmentstatistics(nSEGS, nT1)
        VolumeMetrics = su.segmentationGetVolumemetric(volume_stats)   
        
        # Set the file names
        pnOUT = os.path.join(pn,'Metrics')
        fnFIDSOUT = 'fids_on_%s.fcsv'%(fnroot)
        fnSEGSOUT = 'Segs_on_%s.seg.nrrd'%(fnroot)
        fnVolumeOUT = 'VolumeMetrics_%s.csv'%(fnroot)            
        ffFIDSOUT = su.osnj(pnOUT, fnFIDSOUT)
        ffSEGSOUT = su.osnj(pnOUT, fnSEGSOUT)
        ffVolumeOUT = su.osnj(pnOUT, fnVolumeOUT) 
        VolumeMetrics_forall.append(VolumeMetrics)
        
        # Save the Volume Metrics
        if not flagDemo:
            saveNode(nFIDS, ffFIDSOUT)
            saveNode(nSEGS, ffSEGSOUT) # now we have fiducials on HDD in subject space! 
            su.save_dict_to_csv(VolumeMetrics, ffVolumeOUT)     
            save_listdict_to_csv(VolumeMetrics_forall, summary_path)

        # Save the registration visulization
        if flagSaveVisualization:
            # Capture the whole view for checking registration
            su.setLayout(3)
            su.setVolumeColorLUT(nT1, 'Red')
            su.setVolumeColorLUT(nATL, 'Cyan')
            slicer.util.setSliceViewerLayers(background=nT1, foreground=nATL, label='keep-current', foregroundOpacity=0.5, labelOpacity=None, fit=True)
            su.setSliceOffsetToFiducial('fids_%s'%modality[1],'eyeball_R')
            su.forceViewUpdate() 
            ff_img_out = su.osnj(pnOUT,'imgs_on_%s.png'%(fnroot))
            su.captureImageFromAllViews(ff_img_out)       
    
        if not flagDemo:
            print('Close the Scene')
            su.closeScene()       

namelist = df_fls[0] + df_fls[1]
namelist = [os.path.basename(path) for path in namelist]
save_listdict_to_csv(VolumeMetrics_forall, namelist, summary_path)

#%%



