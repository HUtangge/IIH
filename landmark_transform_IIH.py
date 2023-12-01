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
for iterc, modality in enumerate([['IIH02mm', 'T1'], ['IIH02mm', 'T2']]): 
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
        
        centers_of_eyeballandlens = su.segmentationGetCenterOfMassByRegionName(nSEGS.GetName(), ['R_eyeball', 'R_lens', 'L_eyeball', 'L_lens'])       
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
            scalefactor = 16
            su.setLayout(8)
            su.zoom2D(scalefactor, zoomRed=False, zoomYellow=False)
            su.zoom2D(2, zoomGreen=False)
            FeretdiameterMatrics_single = {}
            
import CrossSectionAnalysis
cs = CrossSectionAnalysis.CrossSectionAnalysisLogic()
centerline = su.getNode('L_ON_Centerlinemodel')
n = getNode('Segs_IIH02mmT2') 
cs.setInputCenterlineNode(centerline)
cs.setLumenSurface(n, 'Segment_15')
            
            
            for idx, region in enumerate(regions_for_centerline): 
                if not re.search('sheath', region['region']):
                    #print('Get the curve node')   
                    optimal_planenormal, point = getOptimalcurve(region['region'], modality[0]+modality[1], 3)
                    outlineModel, closedCurveNode, maxferetlineNode, perpendicularferetlineNode = su.getClosedCurveModel(region['region'], modality[0]+modality[1], optimal_planenormal, point, 3)   
                elif re.search('sheath', region['region']):
                    #print('Get the curve node')
                    optimal_planenormal, point = getOptimalcurve(region['region'], modality[0]+modality[1], 5)
                    outlineModel, closedCurveNode, maxferetlineNode, perpendicularferetlineNode = su.getClosedCurveModel(region['region'], modality[0]+modality[1], optimal_planenormal, point, 5)
                # Extract parameters for the Feret diameter
                keys = ['%s_maxferet_%sPointPosition_%s'%(region['region'], which_point, which_position) for which_point in ['Start', 'End'] for which_position in ['R', 'A', 'S']]
                keys = keys+['%s_perpendicularferet_%sPointPosition_%s'%(region['region'], which_point, which_position) for which_point in ['Start', 'End'] for which_position in ['R', 'A', 'S']]
                values = [value for value in maxferetlineNode.GetCurvePoints().GetPoint(0)+maxferetlineNode.GetCurvePoints().GetPoint(1)]
                values = values+[value for value in perpendicularferetlineNode.GetCurvePoints().GetPoint(0)+perpendicularferetlineNode.GetCurvePoints().GetPoint(1)]
                FeretdiameterMatrics = dict((key,value)for key,value in zip(keys, values))
                maxferetdiameter = su.getDistance(maxferetlineNode.GetCurvePoints().GetPoint(0), maxferetlineNode.GetCurvePoints().GetPoint(1))
                perpendicularferetdiameter = su.getDistance(perpendicularferetlineNode.GetCurvePoints().GetPoint(0), perpendicularferetlineNode.GetCurvePoints().GetPoint(1))
                FeretdiameterMatrics['%s_maxferetdiameter'%(region['region'])] = maxferetdiameter
                FeretdiameterMatrics['%s_perpendicularferetdiameter'%(region['region'])] = perpendicularferetdiameter
                curveareamm2, curvelengthmm = su.getClosedsurfaceareaandlength(closedCurveNode.GetName())
                FeretdiameterMatrics['%s_curveareamm2'%(region['region'])] = curveareamm2
                FeretdiameterMatrics['%s_curvelengthmm'%(region['region'])] = curvelengthmm
                # Check if the diameter plane intersect with the front of the nerve sheath
                if outlinemodel_check.GetPolyData().GetNumberOfPoints() == 0:
                    FeretdiameterMatrics['%s_ifvaliddiameter'%(region['region'])] = 1
                else:
                    FeretdiameterMatrics['%s_ifvaliddiameter'%(region['region'])] = 0
                FeretdiameterMatrics_single.update(FeretdiameterMatrics)
                # Save the Model and Curve
                nameModel = '%s_Diametermodel'%(region['region'])
                nameCurve = '%s_Diametercurve'%(region['region'])
                fnDiametermodelOUT = '%s_%s_%s.vtk'%(nameModel, which_sub, fnroot)
                ffDiametermodelOUT = su.osnj(pnOUT, fnDiametermodelOUT) 
                fnDiametercurveOUT = '%s_%s_%s.mrk.json'%(nameModel, which_sub, fnroot)
                ffDiametercurveOUT = su.osnj(pnOUT, fnDiametercurveOUT) 
                ffperpendicularview = su.osnj(pnOUT, 'img_%s_perpendicularview_%s_%s.png'%(region['region'], which_sub, fnroot))
                ffperpendicularwholeview = su.osnj(pnOUT, 'img_%s_perpendicularwholeview_%s_%s.png'%(region['region'], which_sub, fnroot))
                if not flagDemo:
                    saveNode(outlineModel, ffDiametermodelOUT)
                    saveNode(closedCurveNode, ffDiametercurveOUT)
                # Capture the views
                su.captureImageFromAllViews(ffperpendicularview) 
                # Capture the whole view
                center = (np.array(maxferetlineNode.GetCurvePoints().GetPoint(0)) + np.array(maxferetlineNode.GetCurvePoints().GetPoint(1)))/2
                su.setLayout(3)
                su.setSliceOffsetToFiducial(fidArray = center)
                su.captureImageFromAllViews(ffperpendicularwholeview) 
                su.setLayout(8)
            # Save the extracted parameters for single subject
            fnFeretdiameterMatricsOUT = 'FeretdiameterMatrics_%s_%s.csv'%(which_sub, fnroot)
            ffFeretdiameterMatricsOUT = su.osnj(pnOUT, fnFeretdiameterMatricsOUT) 
            if not flagDemo:
                su.save_dict_to_csv(FeretdiameterMatrics_single, ffFeretdiameterMatricsOUT)       
            # Change the settings back                  
            su.zoom2D(1/scalefactor, zoomRed=False, zoomYellow=False)
            su.setViewtoDefault('Green')                                       

        # Save the Volume Metrics
        if not flagDemo:
            saveNode(nFIDS, ffFIDSOUT)
            saveNode(nSEGS, ffSEGSOUT) # now we have fiducials on HDD in subject space! 
            su.save_dict_to_csv(VolumeMetrics, ffVolumeOUT)     
            su.save_dict_to_csv(VolumeMetrics_forall, summary_path)

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
"""
Tangge: For further parameterization
Add some functions combinations from slicerutil
"""
def getCutterplanes(nameRegion, tag_cohort, number_of_iterations:int = 1000, gap:int=3):
    nameCenterlineModel = '%s_Centerlinemodel'%(nameRegion)
    nerve_tip = su.getFiducialPosition(f'fids_{tag_cohort}', 'nerve_tip_%s'%(re.search('^(.)', nameRegion)[0]))
    centerlinearray = su.modeltopointsarray(nameCenterlineModel)
    dist_to_nervetip = su.getDistance(centerlinearray, nerve_tip) - gap
    point_idx = int(np.where(abs(dist_to_nervetip) == np.min(abs(dist_to_nervetip)))[0])
    direction, point = su.diameterGetplane(nameCenterlineModel, point_idx)
    planenormal_vectors = su.planeGetnormal(direction, number_of_iterations)
    return planenormal_vectors, point

def getClosedCurveArea(nameRegion, tag_cohort, planenormal, point, gap:int = 3):
    plane = su.getPlane(planenormal, point)
    outlinemodel_check = su.diameterGetoutline(f'Seg_{tag_cohort}', '%s_diameterplane_check'%(re.search('^(.)', nameRegion)[0]), plane)
    if outlinemodel_check.GetPolyData().GetNumberOfPoints() == 0:        
        outlinemodel = su.diameterGetoutline(f'Seg_{tag_cohort}', '%s_optic_nerve_sheath_anterior_with_nerve'%(re.search('^(.)', nameRegion)[0]), plane)        
        if outlinemodel.GetPolyData().GetNumberOfLines() == 1:
            outlinearray = su.modeltopointsarray(outlinemodel.GetName())
            dist_arr = su.getDistance(outlinearray, outlinearray, metric='euclidean')
            closedCurveNode = su.getSortedcurvepoints(outlinearray[0,:], outlinearray, f'{gap}mmDiametercurve')
            su.getResampledcurve(closedCurveNode.GetName(), 1)
            curveareamm2 = slicer.modules.markups.logic().GetClosedCurveSurfaceArea(closedCurveNode)
            slicer.mrmlScene.RemoveNode(closedCurveNode)
        else:
            curveareamm2 = 9999
        slicer.mrmlScene.RemoveNode(outlinemodel)
    else:
        curveareamm2 = 9999
    slicer.mrmlScene.RemoveNode(outlinemodel_check)
    return curveareamm2

def remove_invalid_intersection_area(intersection_area):
    while np.min(intersection_area) < 15:
        intersection_area[np.where(intersection_area == np.min(intersection_area))[0][0]] = 9999
    return intersection_area

def getOptimalcurve(nameRegion, tag_cohort, number_of_iterations, gap:int = 3):
    planenormal_vectors, point = getCutterplanes(nameRegion, tag_cohort, number_of_iterations, gap)
    intersection_area = []
    for i in range(len(planenormal_vectors)):
        curveareamm2 = getClosedCurveArea(nameRegion, tag_cohort, planenormal_vectors[i,:], point, gap)    
        intersection_area.append(curveareamm2)
    intersection_area = remove_invalid_intersection_area(intersection_area)
    percentage_of_valid_iteration = 1 - intersection_area.count(9999)/len(intersection_area)
    optimal_planenormal = planenormal_vectors[np.where(intersection_area == np.min(intersection_area))[0][0]]    
    return optimal_planenormal, point, percentage_of_valid_iteration

#%% RUN IN SLICER
# Set the file directory
ptnCenterline = '*Centerlinemodel*.vtk' # Cosmo1mm / Astronauts / Cosmonauts
ptnSEGS = 'Segs*.seg.nrrd' # Way to use: name = fnSEGSptn %(a,b)
ptnFIDS = 'fids*.fcsv'
pnMRIptn = r'D:\GeTang\SANS\Raw_data\Cosmonaut_BIDS'
ptnMRI = 'Denoised_*_T1w.nii'
regions_for_centerline = [{'region': 'L_optic_nerve', 'fids': 'L_optic_nerve_endpoints'},
                          {'region': 'R_optic_nerve', 'fids': 'R_optic_nerve_endpoints'},
                          {'region': 'L_optic_nerve_sheath_anterior_with_nerve', 'fids': 'L_optic_nerve_sheath_endpoints'}, 
                          {'region': 'R_optic_nerve_sheath_anterior_with_nerve', 'fids': 'R_optic_nerve_sheath_endpoints'}]

# Measure the sheath
sheathDiameter = True
# Save visulization
flagSaveVisualization = True
views3D = ['left', 'right', 'superior', 'anterior']

# Try something new
flagDemo = False
df_fls = []
FeretdiameterMatrics_forall = []
# for each cohort
for iterc, tag_cohort in enumerate(['Cosmo02mm']): 
    print(iterc, tag_cohort)    
    pnMRI  = pnMRIptn    
    # Attention: Locate file use fnmatch, which is different from regular expression
    fl = fs.locateFiles(ptnMRI, pnMRI, level=3)
    df_fls.append(fl)
        
    # for each file
    for idx, ff in enumerate(fl):
        if idx in [32, 34, 55]:
            ff = ff.replace('/','\\')
            logging.info(f'Processing {idx} {ff}')
            su.setLayout(3)
            
            if flagDemo:
                if idx == 1:
                    print('End of processing')
                    time.sleep(3)
                    break  
                
            # load T1 volume
            success,nT1 = loadVolume(ff,returnNode=True)    
            
            # extract fn root: sub-cosmonaut01_ses-postflight_T1w_n4
            pn, fn = os.path.split(ff)
            pn = r'{}'.format(pn)
            fnroot, fnext = fs.splitext(fn)        
            which_sub = re.search('sub-(.+?)_ses', fnroot).group(1)
            pnOUT = os.path.join(pn,'SANS')
            pnOUToptimal = os.path.join(pn,'SANSoptimal')
            if not os.path.exists(pnOUToptimal):
                os.mkdir(pnOUToptimal)
            flsegs = fs.locateFiles(ptnSEGS, pnOUT, level=1)[0].replace('/','\\')
            success,nSEGS = loadSegmentation(flsegs, returnNode=True)    
            nSEGS.SetName(f'Seg_{tag_cohort}')
            flfids = fs.locateFiles(ptnFIDS, pnOUT, level=1)[0].replace('/','\\')
            nFIDS = loadMarkupsFiducialList(flfids, returnNode=True)
            nFIDS.SetName(f'fids_{tag_cohort}')
            
            # Load the Centerline Model
            flcenterline = fs.locateFiles(ptnCenterline, pnOUT, level=1)
            for i in range(4):
                flcenterline[i] = flcenterline[i].replace('/','\\')
                success,ncenterline = loadModel(flcenterline[i], returnNode=True)
                nameCenterline = '%sCenterlinemodel'%(ncenterline.GetName().split('Centerlinemodel')[0])
                ncenterline.SetName(nameCenterline)
                if re.search('sheath', nameCenterline):
                    slicer.mrmlScene.RemoveNode(ncenterline)
                    
            # Make some segment invisible for later 3D visulization
            segmentationDisplayNode = nSEGS.GetDisplayNode()
            L_diameterplane_check_ID = nSEGS.GetSegmentation().GetSegmentIdBySegmentName('L_diameterplane_check')
            R_diameterplane_check_ID = nSEGS.GetSegmentation().GetSegmentIdBySegmentName('R_diameterplane_check')
            segmentationDisplayNode.SetSegmentOpacity(L_diameterplane_check_ID, 0.0)
            segmentationDisplayNode.SetSegmentOpacity(R_diameterplane_check_ID, 0.0)
            # Save sheath information
            if sheathDiameter:  
                #print('Processing sheathdiameter')
                scalefactor = 16
                su.setLayout(8)
                su.zoom2D(scalefactor, zoomRed=False, zoomYellow=False)
                su.zoom2D(2, zoomGreen=False)
                FeretdiameterMatrics_single = {}
                for idx, region in enumerate(regions_for_centerline): 
                    if not re.search('sheath', region['region']):
                        #print('Get the curve node')   
                        start = time.time()
                        optimal_planenormal, point, percentage_of_valid_iteration = getOptimalcurve(region['region'], tag_cohort, 2000, 3)
                        outlineModel, closedCurveNode, maxferetlineNode, perpendicularferetlineNode = su.getClosedCurveModel(region['region'], tag_cohort, optimal_planenormal, point, 3)   
                        logging.info(f'{time.time()-start}')                
                        # Extract parameters for the Feret diameter
                        keys = ['%s_maxferet_%sPointPosition_%s'%(region['region'], which_point, which_position) for which_point in ['Start', 'End'] for which_position in ['R', 'A', 'S']]
                        keys = keys+['%s_perpendicularferet_%sPointPosition_%s'%(region['region'], which_point, which_position) for which_point in ['Start', 'End'] for which_position in ['R', 'A', 'S']]
                        values = [value for value in maxferetlineNode.GetCurvePoints().GetPoint(0)+maxferetlineNode.GetCurvePoints().GetPoint(1)]
                        values = values+[value for value in perpendicularferetlineNode.GetCurvePoints().GetPoint(0)+perpendicularferetlineNode.GetCurvePoints().GetPoint(1)]
                        FeretdiameterMatrics = dict((key,value)for key,value in zip(keys, values))
                        maxferetdiameter = su.getDistance(maxferetlineNode.GetCurvePoints().GetPoint(0), maxferetlineNode.GetCurvePoints().GetPoint(1))
                        perpendicularferetdiameter = su.getDistance(perpendicularferetlineNode.GetCurvePoints().GetPoint(0), perpendicularferetlineNode.GetCurvePoints().GetPoint(1))
                        FeretdiameterMatrics['%s_maxferetdiameter'%(region['region'])] = maxferetdiameter
                        FeretdiameterMatrics['%s_perpendicularferetdiameter'%(region['region'])] = perpendicularferetdiameter
                        curveareamm2, curvelengthmm = su.getClosedsurfaceareaandlength(closedCurveNode.GetName())
                        FeretdiameterMatrics['%s_curveareamm2'%(region['region'])] = curveareamm2
                        FeretdiameterMatrics['%s_curvelengthmm'%(region['region'])] = curvelengthmm
                        FeretdiameterMatrics['%s_percentageofvaliditeration'%(region['region'])] = percentage_of_valid_iteration
                        FeretdiameterMatrics_single.update(FeretdiameterMatrics)                
                        # Save the Model and Curve
                        nameModel = '%s_optimal_Diametermodel'%(region['region'])
                        nameCurve = '%s_optimal_Diametercurve'%(region['region'])
                        fnDiametermodelOUT = '%s_%s_%s.vtk'%(nameModel, which_sub, fnroot)
                        ffDiametermodelOUT = su.osnj(pnOUToptimal, fnDiametermodelOUT) 
                        fnDiametercurveOUT = '%s_%s_%s.mrk.json'%(nameModel, which_sub, fnroot)
                        ffDiametercurveOUT = su.osnj(pnOUToptimal, fnDiametercurveOUT) 
                        ffperpendicularview = su.osnj(pnOUToptimal, 'img_optimal_%s_perpendicularview_%s_%s.png'%(region['region'], which_sub, fnroot))
                        ffperpendicularwholeview = su.osnj(pnOUToptimal, 'img_optimal_%s_perpendicularwholeview_%s_%s.png'%(region['region'], which_sub, fnroot))
                        saveNode(outlineModel, ffDiametermodelOUT)
                        saveNode(closedCurveNode, ffDiametercurveOUT)
                        # Capture the views
                        su.captureImageFromAllViews(ffperpendicularview) 
                        # Capture the whole view
                        center = (np.array(maxferetlineNode.GetCurvePoints().GetPoint(0)) + np.array(maxferetlineNode.GetCurvePoints().GetPoint(1)))/2
                        su.setLayout(3)
                        su.setSliceOffsetToFiducial(fidArray = center)
                        su.captureImageFromAllViews(ffperpendicularwholeview) 
                        su.setLayout(8)
                # Save the extracted parameters for single subject
                fnFeretdiameterMatricsOUT = 'FeretdiameterMatrics_optimal_%s_%s.csv'%(which_sub, fnroot)
                ffFeretdiameterMatricsOUT = su.osnj(pnOUToptimal, fnFeretdiameterMatricsOUT) 
                su.save_dict_to_csv(FeretdiameterMatrics_single, ffFeretdiameterMatricsOUT)       
                # Change the settings back                  
                su.zoom2D(1/scalefactor, zoomRed=False, zoomYellow=False)
                su.setViewtoDefault('Green')                                               
            # Save the registration visulization
            if flagSaveVisualization:
                # Set parameters for the view to save
                su.setLayout(4)
                su.view3D_center()
                su.setOpacityforSegmentation(0.3)
                segmentationDisplayNode.SetSegmentOpacity(L_diameterplane_check_ID, 0.0)
                segmentationDisplayNode.SetSegmentOpacity(R_diameterplane_check_ID, 0.0)        
                su.threeDViewSetBackgroundColor('white')
                su.setFiducialVisibility(False) 
                su.zoom3D(3)
                for view in views3D:
                    su.view3D_lookFromViewAxis(view)
                    time.sleep(3)
                    su.captureImageFromAllViews(su.osnj(pnOUToptimal, f'{view}_optimal_3Dview_{which_sub}.png'))   
                    time.sleep(3)
        
            FeretdiameterMatrics_forall.append(FeretdiameterMatrics_single)
            segmentationDisplayNode.SetAllSegmentsOpacity(1)
            del segmentationDisplayNode
            
            if not flagDemo:
                print('Close the Scene')
                su.closeScene()       

su.save_dict_to_csv(FeretdiameterMatrics_forall, 'D://GeTang//SANS//Raw_data//Summary//FeretdiameterMatrics_forall_05052021.csv')



