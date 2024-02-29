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
    print(type(keys))
    print(keys)
    keys.add('filename')
    keys.add('modality')
    search_for = 'optic_nerve'
    replace_with = 'ON'
    # Open a CSV file for writing
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        # Create a DictWriter object with the determined keys
        writer = csv.DictWriter(file, fieldnames=keys)
        # Write the header (field names)
        writer.writeheader()
        # Write the dictionary data
        for row, item in zip(data, namelist):
            row = {key.replace(search_for, replace_with): value for key, value in row.items()}
            if item == 'Denoised_sub-02_ses-01_T1w.nii':
                print(len(row.keys()))
            item = {'filename':item}
            if len(row) < 150:
                modality = {'modality':'T1'}
            else:
                modality = {'modality':'T2'}
            row.update(item)
            row.update(modality)
            writer.writerow(row)

def LabelmapHardenTransform(namesLabelmap:list, transform):
    for name in namesLabelmap:
        nlab = getNode(name)
        nlab.SetAndObserveTransformNodeID(transform.GetID())
        nlab.HardenTransform()


"""
Tangge: This is the code for 20231127
Rerun to get the volumn of the eyeball and the individualized center of eyeball and lens
"""
#%% RUN IN SLICER 
# Configurations
project_path = r'D:\users\getang\IIH'
flagDemo = False
centerline = True
regions_for_centerline = [{'region': 'L_ON', 'fids': 'L_ON_endpoints'},
                          {'region': 'R_ON', 'fids': 'R_ON_endpoints'}]
# Measure the sheath
sheathDiameter = False
# Save visulization
flagSaveVisualization = True # Save the image
views3D = ['left', 'right', 'superior', 'anterior']

# Naming the destination for the transformers and the segmentations
pnATL = r'%s\template\High_resolution_template'%project_path
fnATLptn = 'T_%s%s.nii.gz'
pnFIDS = r'%s\data\Segmentations'%project_path
fnSEGSptn = r'Segs_%s%s.seg.nrrd' # Way to use: name = fnSEGSptn %(a,b)
fnFIDSptn = r'fids_%s%s.fcsv'
pnMRI = r'%s\data\Rawdata'%project_path
fnMRIptn = r'Denoised_*_%sw.nii'
summary_path = r'%s\data\Rawdata\Summary\volumetrics_forall_IIH_30112023.csv'%project_path
# Set the log file for later check 
logging.basicConfig(filename=r'%s\data\Rawdata\Summary\log_scriptAstronaut_landmark_transform.log'%project_path, encoding='utf-8', level=logging.DEBUG)

df_fls = []
VolumeMetrics_forall = []
FeretdiameterMatrics_forall = []
#%% The main loop
# for each cohort
# for iterc, modality in enumerate([['IIH02mm', 'T1'], ['IIH02mm', 'T2']]): 
for iterc, modality in enumerate([['IIH02mm', 'T1']]): 
    print(iterc, modality)    
    fnATL  = fnATLptn%(modality[0],modality[1])
    fnSEGS = fnSEGSptn%(modality[0],modality[1])
    fnFIDS = fnFIDSptn%(modality[0],modality[1])
    fnMRI = fnMRIptn%(modality[1])

    # Attention: Locate file use fnmatch, which is different from regular expression
    fl = fs.locateFiles(fnMRI, pnMRI, level=1)
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

        # Convert vtkMRMLSegmentationNode to vtkMRMLLabelMapVolumeNode for apply transform
        fnSEGSroot = os.path.basename(os.path.join(pnFIDS,fnSEGS))[:os.path.basename(os.path.join(pnFIDS,fnSEGS)).find('.')]
        fnATLroot = os.path.basename(os.path.join(pnATL,fnATL))[:os.path.basename(os.path.join(pnATL,fnATL)).find('.')]
        su.segmentationExportToIndividualLabelmap(fnSEGSroot, fnATLroot) 
        nSEGSlabelnames, _ = su.segmentationListRegions(fnSEGSroot)
        nSEGSlabelnames = [element + "_labelmap" for element in nSEGSlabelnames]
        # Export to models instead of the labelmap
        model_FolderItemId = su.segmentationExportToModelsByRegionNames(fnSEGSroot, f"{fnSEGSroot}_model")
        slicer.mrmlScene.RemoveNode(nSEGS)

        # Load the transforms
        success, nTrfATLtoT1_AFF = slicer.util.loadTransform(os.path.join(pn,fnAFF), returnNode=True)
        success, nTrfATLtoT1_DEF = slicer.util.loadTransform(os.path.join(pn,fnDEF), returnNode=True)        
        nTrfATLtoT1_AFF.SetAndObserveTransformNodeID(nTrfATLtoT1_DEF.GetID())
        nATL.SetAndObserveTransformNodeID(nTrfATLtoT1_AFF.GetID())
        nFIDS.SetAndObserveTransformNodeID(nTrfATLtoT1_AFF.GetID())        
        # nSEGS.SetAndObserveTransformNodeID(nTrfATLtoT1_AFF.GetID())       

        # Save the transformed segmentation and fiducials
        nFIDS.HardenTransform() # IMPORTANT!!
        # nSEGS.HardenTransform() # IMPORTANT!!     
        # nSEGSlabel.HardenTransform()
        LabelmapHardenTransform(nSEGSlabelnames, nTrfATLtoT1_AFF)
        su.transformApplytransformtoModelFolder(model_FolderItemId, nTrfATLtoT1_AFF)

        nSEGSnewModel = su.segmentationImportModelsInFolder(model_FolderItemId, f"{fnSEGSroot}_model")
        slicer.mrmlScene.RemoveNode(model_FolderItemId)

        nSEGS = nSEGSnewModel
        
        centers_of_eyeballandlens = su.segmentationGetCenterOfMassByRegionName(nSEGS.GetName(), ['R_eyeball', 'R_lens', 'L_eyeball', 'L_lens'])       
        break
        su.fiducialListFromArray(nFIDS.GetName(), centers_of_eyeballandlens, ['individualized_center_R_eyeball', 'individualized_center_R_lens', 'individualized_center_L_eyeball', 'individualized_center_L_lens'])
        su.visFid_SetVisibility(nFIDS, locked = True, visibility = False)
        
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
        
        saveNode(nSEGS, f"D:/users/getang/IIH/data/Rawdata/sub-02_ses-01/test/{fnSEGSOUT}")
        # Save centerline model
        if centerline: 
            for region in regions_for_centerline: 
                # Set the filenames
                nameModel = '%s_Centerlinemodel'%(region['region'])
                nameTable = '%s_CenterlineMetrics'%(region['region'])
                fnCenterlineOUT = '%s_%s_%s.vtk'%(nameModel, which_sub, fnroot) 
                fnCenterlineMetricsOUT = '%s_%s_%s.csv'%(nameTable, which_sub, fnroot)
                if not re.search('sheath', region['region']):                        
                    name_of_fids = ['nerve_tip_%s'%(re.search('^(.)', region['region'])[0]), '%s_start_of_bone_part'%(region['region'])]
                    endpoints = su.arrayFromFiducialList(nFIDS.GetName(), name_of_fids)
                    fids_endpoints = su.fiducialListFromArray(region['fids'], endpoints, name_of_fids)
                    centerlinePolyData, _ = su.centerlineExtractcenterline(nSEGS.GetName(), region['region'], region['fids'], preprocesssurface = True)
                    if centerlinePolyData.GetNumberOfPoints() < 10:
                        logging.info(f'{ff} Centerline without preprocessing')
                        centerlinePolyData, _ = su.centerlineExtractcenterline(nSEGS.GetName(), region['region'], region['fids'], preprocesssurface = False)
                    centerlineModel = su.centerlineGetmodel(centerlinePolyData, 1, nameModel)                 
                elif re.search('sheath', region['region']):    
                    name_of_fids = ['nerve_tip_%s'%(re.search('^(.)', region['region'])[0]), '%s_optic_nerve_sheath_endpoint'%(re.search('^(.)', region['region'])[0])]
                    endpoints = su.arrayFromFiducialList(nFIDS.GetName(), name_of_fids)
                    fids_endpoints = su.fiducialListFromArray(region['fids'], endpoints, name_of_fids)
                    centerlinePolyData, _ = su.centerlineExtractcenterline(nSEGS.GetName(), region['region'], region['fids'], preprocesssurface = True)
                    if centerlinePolyData.GetNumberOfPoints() < 10:
                        logging.info(f'{ff} Centerline without preprocessing')
                        centerlinePolyData, _ = su.centerlineExtractcenterline(nSEGS.GetName(), region['region'], region['fids'], preprocesssurface = False)
                    centerlineModel = su.centerlineGetmodel(centerlinePolyData, 0, nameModel)
                centerlineMetrics = su.centerlineGetcenterlineproperties(centerlinePolyData, nameTable)
                # Set file names
                ffCenterlineOUT = su.osnj(pnOUT, fnCenterlineOUT) 
                ffCenterlineMetricsOUT = su.osnj(pnOUT, fnCenterlineMetricsOUT)                  
                if not flagDemo:
                    saveNode(centerlineModel, ffCenterlineOUT)
                    su.save_tablepolydata_to_csv(centerlineMetrics, ffCenterlineMetricsOUT)                

        # Save sheath information
        if sheathDiameter and modality[1] == 'T2':  
            import CrossSectionAnalysis
            for idx, region in enumerate(regions_for_centerline): 
                csa = CrossSectionAnalysis.CrossSectionAnalysisLogic()
                outputTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
                centerline = su.getNode(r'%s_Centerlinemodel'%region['region'])
                csa.setInputCenterlineNode(centerline)
                csa.setLumenSurface(nSEGS, nSEGS.GetSegmentation().GetSegmentIdBySegmentName(r'%sS'%region['region']))
                csa.setOutputTableNode(outputTableNode)
                csa.updateOutputTable(csa.inputCenterlineNode, csa.outputTableNode)
                csatable = csa.outputTableNode
                fnsheathdiameterMetricsOUT = '%s_SheathMetrics_%s_%s.csv'%(region['region'], which_sub, fnroot)
                ffsheathdiameterMetricsOUT = su.osnj(pnOUT, fnsheathdiameterMetricsOUT) 
                # Save the Model and Curve
                if not flagDemo:
                    su.save_tablepolydata_to_csv(csatable, ffsheathdiameterMetricsOUT)

        # Save the Volume Metrics
        if not flagDemo:
            saveNode(nFIDS, ffFIDSOUT)
            saveNode(nSEGS, ffSEGSOUT) # now we have fiducials on HDD in subject space! 
            su.save_dict_to_csv(VolumeMetrics, ffVolumeOUT)     

        # Save the registration visulization
        if flagSaveVisualization:
            # Capture the whole view for checking registration
            su.setLayout(3)
            su.setVolumeColorLUT(nT1, 'Red')
            su.setVolumeColorLUT(nATL, 'Cyan')
            slicer.util.setSliceViewerLayers(background=nT1, foreground=nATL, label='keep-current', foregroundOpacity=0.5, labelOpacity=None, fit=True)
            su.setSliceOffsetToFiducial('fids_%s%s'%(modality[0],modality[1]),'R_eyeball')
            su.forceViewUpdate() 
            ff_img_out = su.osnj(pnOUT,'imgs_on_%s.png'%(fnroot))
            su.captureImageFromAllViews(ff_img_out)       
    
        if not flagDemo:
            print('Close the Scene')
            su.closeScene()       

namelist = df_fls[0] + df_fls[1]
namelist = [os.path.basename(path) for path in namelist]
save_listdict_to_csv(VolumeMetrics_forall, namelist, summary_path)
