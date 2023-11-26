# -*- coding: utf-8 -*-
"""
This is for get the transforms of the raw image to template
"""

#%% scriptAstronautCT.py
import os
from os.path import join as osj
from os.path import normpath as osn
import re

import sys
sys.path.append(r"D:\users\getang\SANS\Slicertools")
import file_search_tool as fs
#from utils import baseutils as bu
from importlib import reload
import time
import subprocess

from nipype.interfaces import ants
from subprocess import check_output
import subprocess
import shutil
import SimpleITK as sitk

#%% helper functions
def regAnts(ffFixed,ffMoving,output_prefix='output_',
            transforms=['Affine', 'SyN'],
            transform_parameters=[(2.0,), (0.25, 3.0, 0.0)],
            number_of_iterations=[[1500, 200], [70, 90, 20]],
            metric=['Mattes','CC'],
            metric_weight=[1,1],
            num_threads = 4,
            radius_or_number_of_bins = [32,2],
            sampling_strategy = ['Random', None],
            sampling_percentage = [0.05, None],
            convergence_threshold = [1.e-8, 1.e-9],
            convergence_window_size = [20,20],
            smoothing_sigmas = [[2,1], [3,2,1]],
            sigma_units = ['vox','vox'],
            composite_trf = False,
            shrink_factors = [[4,2], [4,3,2]],
            use_histogram_matching = [True, True],
            initial_geometric_Align = False,
            verbose = False,
            output_warped_image = 'output_warped_image.nii.gz'):
    reg = ants.Registration()
    reg.inputs.fixed_image = ffFixed
    reg.inputs.moving_image = ffMoving
    reg.inputs.output_transform_prefix = output_prefix
    #reg.inputs.initial_moving_transform = 'trans.mat'
    #reg.inputs.invert_initial_moving_transform = True
    reg.inputs.transforms = transforms
    reg.inputs.transform_parameters = transform_parameters
    reg.inputs.number_of_iterations = number_of_iterations
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = False
    reg.inputs.collapse_output_transforms = composite_trf
    reg.inputs.metric = metric
    reg.inputs.metric_weight = metric_weight # Default (value ignored currently by ANTs)
    reg.inputs.num_threads = num_threads
    reg.inputs.radius_or_number_of_bins = radius_or_number_of_bins
    reg.inputs.sampling_strategy = sampling_strategy
    reg.inputs.sampling_percentage = sampling_percentage
    reg.inputs.convergence_threshold = convergence_threshold
    reg.inputs.convergence_window_size = convergence_window_size
    reg.inputs.smoothing_sigmas = smoothing_sigmas
    reg.inputs.sigma_units = sigma_units
    reg.inputs.shrink_factors = shrink_factors
    reg.inputs.use_histogram_matching = use_histogram_matching
    reg.inputs.output_warped_image = output_warped_image
    #reg.inputs.output_inverse_warped_image = output_inverse_warped_image
    if initial_geometric_Align:
        reg.inputs.initial_moving_transform_com = 0
    if verbose:
        reg.inputs.verbose = True
    print(reg.cmdline)
    return reg

def n4normalize(ffIN,ffOUT):
    n4 = ants.N4BiasFieldCorrection(dimension=3)
    n4.inputs.input_image = ffIN
    n4.inputs.output_image = ffOUT
    print(n4.cmdline)
    return n4

def run_n4normalize(ffIN,ffOUT,runcmd=True,skipIfExists=True):
    n4 = n4normalize(ffIN,ffOUT)
    if runcmd:
        if skipIfExists:
            if not os.path.exists(ffOUT):
                subprocess.check_output(n4.cmdline, shell=True)
            else:
                print('N4 correction already computed (N4 already present):\n%s'%ffOUT)
        else:
            subprocess.check_output(n4.cmdline, shell=True)
    return n4
    
def run_reg(ffFIX,ffMOV,
            output_prefix = 'out_', 
            output_warped_image= 'deformed.nii',
            pnOUT=None,ffAFF=None,ffDEF=None,ffAFFinv=None,
            ffDEFinv=None,DEFVOL=None,
            skipIfExists=True,runcmd=True):    
    pnFIX, fnFIX = os.path.split(ffFIX)
    pnMOV, fnMOV = os.path.split(ffMOV)
    
    # if pnOUT is None:
    #     pnOUT = pnFIX
    
    fnFIX_root, fnFIX_ext = splitext(fnFIX)
    fnMOV_root, fnMOV_ext = splitext(fnMOV)
    # which_sub = re.search('sub-(.+?)_ses', fnFIX_root)
    # '''
    # The regexp treat '-' as a separation for words, So try to separate each step with '-' instead of '_'
    # '''
    # if which_sub:
    #     sub_root = fnFIX_root
    #     trfPrefix = osnj(pnOUT,'SANS',f'trf-Temp_to_{which_sub.group(1)}')
    # else:
    #     sub_root = fnMOV_root
    #     which_sub = re.search('sub-(.+?)_ses', fnMOV_root).group(1)
    #     trfPrefix = osnj(pnOUT,'SANS',f'trf-Temp_to_{which_sub.group(1)}')
    sub_root = 'test'
    pnOUT = r'D:\users\getang\IIH\test'
    trfPrefix = osnj(pnOUT, 'trf_%s_to_%s'%(fnMOV_root,fnFIX_root))
    
    if ffAFF is None:
        ffAFF = osnj(pnOUT,f'{trfPrefix}_AFF-{sub_root}.mat')
        
    if ffDEF is None:
        ffDEF = osnj(pnOUT,f'{trfPrefix}_DEF-{sub_root}.nii.gz')
        
    if DEFVOL is None:
        ffDEFVOL = osnj(pnOUT,f'{trfPrefix}_DEFVOL-{sub_root}.nii.gz')
    
    if ffAFFinv is None:
        ffAFFinv = osnj(pnOUT,f'{trfPrefix}_AFFinv-{sub_root}.mat')
        
    if ffDEFinv is None:
        ffDEFinv = osnj(pnOUT,f'{trfPrefix}_DEFinv-{sub_root}.nii.gz')
            
    # if DEFVOLinv is None:
    #     ffDEFVOLinv = osnj(pnOUT,f'{trfPrefix}_DEFVOLinv-{sub_root}.nii.gz')
    
    reg = regAnts(ffFIX,ffMOV,
                  output_prefix = output_prefix, 
                  output_warped_image=output_warped_image,
                  transform_parameters=[(0.5,), (0.25, 3.0, 0.0)],
                  smoothing_sigmas = [[2,1], [4,2,1,0]],
                  shrink_factors = [[4,2], [8,4,2,1]],
                  number_of_iterations=[[500, 250], [100, 70, 50, 40]],
                  composite_trf = True,
                  convergence_threshold = [1.e-7, 1.e-8],
                  verbose = True,
                  initial_geometric_Align=True)
    
    
    if runcmd:
        if skipIfExists:
            if not os.path.exists(ffDEF):
                linux_fit_cmd = 'wsl' + ' ' + convert_cmd_to_linux(reg.cmdline)
                subprocess.check_output(linux_fit_cmd, shell=True)
                # after registration, rename files and create inv affine trf
                outpfx = reg.inputs.output_transform_prefix
                ffoutAFF        = outpfx + '0GenericAffine.mat'
                ffoutDEF        = outpfx + '1Warp.nii.gz'
                ffoutDEFinv     = outpfx + '1InverseWarp.nii.gz'
                ffoutDEFVOL     = outpfx + 'volATLDeformed.nii.gz'
                #ffoutDEFVOLinv  = outpfx + 'volATLDeformedinv.nii.gz'
                                
                # move files
                shutil.move(convert_linux_to_win(ffoutAFF),ffAFF)
                shutil.move(convert_linux_to_win(ffoutDEF),ffDEF)
                shutil.move(convert_linux_to_win(ffoutDEFinv),ffDEFinv)
                shutil.move(convert_linux_to_win(ffoutDEFVOL),ffDEFVOL)
                #shutil.move(convert_linux_to_win(ffoutDEFVOLinv),ffDEFVOLinv)
                # invert affine matrix and store
                trf = sitk.ReadTransform(ffAFF)
                trfinv = trf.GetInverse()
                sitk.WriteTransform(trfinv,ffAFFinv)
            else:
                print('Transformation already computed (def-field already present):\n%s'%ffDEF)
        else:
            linux_fit_cmd = 'wsl' + ' ' + convert_cmd_to_linux(reg.cmdline)
            subprocess.check_output(linux_fit_cmd, shell=True)
            # after registration, rename files and create inv affine trf
            outpfx = reg.inputs.output_transform_prefix
            ffoutAFF        = outpfx + '0GenericAffine.mat'
            ffoutDEF        = outpfx + '1Warp.nii.gz'
            ffoutDEFinv     = outpfx + '1InverseWarp.nii.gz'
            ffoutDEFVOL     = outpfx + 'volATLDeformed.nii.gz'
            #ffoutDEFVOLinv  = outpfx + 'volATLDeformedinv.nii.gz'
                            
            # move files
            shutil.move(convert_linux_to_win(ffoutAFF),ffAFF)
            shutil.move(convert_linux_to_win(ffoutDEF),ffDEF)
            shutil.move(convert_linux_to_win(ffoutDEFinv),ffDEFinv)
            shutil.move(convert_linux_to_win(ffoutDEFVOL),ffDEFVOL)
            #shutil.move(convert_linux_to_win(ffoutDEFVOLinv),ffDEFVOLinv)
            # invert affine matrix and store
            trf = sitk.ReadTransform(ffAFF)
            trfinv = trf.GetInverse()
            sitk.WriteTransform(trfinv,ffAFFinv)
    return reg

def splitext(fn):
    if fn.endswith('.nii.gz'):
        root = fn.replace('.nii.gz','')
        ext = '.nii.gz'
        return (root, ext)
    else:
        return os.path.splitext(fn)

def osnj(*args):
    return osn(osj(*args)).replace('\\','/')

def convert_cmd_to_linux(cmd):
    return cmd.replace('D:', '/mnt/d').replace('\\','/')                       

def convert_linux_to_win(filename:str):
    return filename.replace('/mnt/d', 'D:').replace('/','\\')

#%% Registration
# !!! For the T1w and T2w images, the Atalas is the moving image. 
# Reasoning behind this is we register the larger image to the smaller image
proj_path = r'D:\users\getang\IIH'
project_name = 'IIH'

# Template path
fnATL = 'T_template1.nii.gz'
ffATL = osn(osj(proj_path, 'test', fnATL))

# Raw data path
ffEye = 'Denoised_sub-15_ses-01_T2w.nii'
ffT2 = osn(osj(proj_path, 'test', ffEye))

#%%
linux_proj_path = convert_cmd_to_linux(osj(proj_path, 'test'))    
# subprocess.check_output(['wsl', 'mkdir', linux_proj_path])
prefixOUT = f'{linux_proj_path}/out_'
ffOUT = f'{linux_proj_path}/out_volATLDeformed.nii.gz'

#%%
ffMOV = ffT2
ffFIX = ffATL
t0 = time.time()
reg = run_reg(ffFIX,ffMOV,
              output_prefix = prefixOUT, 
              output_warped_image=ffOUT)
              
t1 = time.time()

print('Done')

#%%
# =============================================================================
# for i in range(flT1.shape[0]):
#     #if i>2:
#     #    break
#     ffT1 = bu.osnj(flT1.ff[i])
#     ffT1N4 = ffT1.replace('.nii','_n4.nii.gz')
#     t0 = bu.tic()
#     n4 = run_n4normalize(ffT1,ffT1N4)
#     print('\n\nFinished N4 correction: %d of %d (%s)\nElapsed time: %0.3f sec.\n\n'%
#           (i+1,flT1.shape[0],flT1.fn[i],bu.toc(t0)))
#     
# =============================================================================
#% ANTS registration of each N4 normalized image to template
# sys.stdout.write = open("D:\GeTang\SANS\log_files\log_test.txt", "a")

for i in range(flT1.shape[0]):
    # if i>2:
    #     break
    if i == 81:
        ffT1  = osnj(flT1.ff[i])
        
        # set up project path under each folder
        pnffT1, fnffT1 = os.path.split(ffT1)   
        proj_path = f'{pnffT1}/{project_name}'
        linux_proj_path = convert_cmd_to_linux(proj_path)     
        subprocess.check_output(['wsl', 'mkdir', linux_proj_path])
        prefixOUT = f'{linux_proj_path}/out_'
        ffOUT = f'{linux_proj_path}/out_volATLDeformed.nii.gz'
        #ffOUTinv = f'{linux_proj_path}/out_volATLDeformedinv.nii.gz'
        
        ffMOV = ffATL
        ffFIX = ffT1
        t0 = time.time()
        reg = run_reg(ffFIX,ffMOV,
                      output_prefix = prefixOUT, 
                      output_warped_image=ffOUT)
                      #output_inverse_warped_image = ffOUTinv)
        t1 = time.time()
        print('\n\nFinished ANTS reg: %d of %d (%s)\nElapsed time: %0.3f sec.\n\n'%
              (i+1,flT1.shape[0],flT1.fn[i],(t1-t0)))

"""
Tang Ge : Attention !!!
There is issue with astronauts for the deformed registration, 
The index for the subjects are : 4, 25, 72, 73, 74, 75, 79, 80, 95 
"""

# reg = regAnts(ffFIX,ffMOV,
#                 output_prefix = prefixOUT, 
#                 output_warped_image=ffOUT,
#                 transform_parameters=[(0.5,), (0.25, 3.0, 0.0)],
#                 smoothing_sigmas = [[2,1], [4,2,1,0]],
#                 shrink_factors = [[4,2], [8,4,2,1]],
#                 number_of_iterations=[[500, 250], [100, 70, 50, 40]],
#                 composite_trf = True,
#                 convergence_threshold = [1.e-7, 1.e-8],
#                 verbose = True,
#                 initial_geometric_Align=True)
# sys.stdout.close()
#%% 2019-03-26 register the previous landmark-atlas to the two new templates

# if __name__ == "__main__":
#     ffATLold = bu.osnj(r'D:\Dropbox\Projects\AstronautT1\data\prelimTemplateAtlasTagging_2019\template_AstronautsT1\T_template_ForTagging2019.nii.gz')
#     ffATL_Astro_new = bu.osnj(r'D:\Dropbox\Projects\AstronautT1\data\finalTemplatesJournal_2019\T_template_AstronautsT1.nii.gz')
#     ffATL_Cosmo_new = bu.osnj(r'D:\Dropbox\Projects\AstronautT1\data\finalTemplatesJournal_2019\T_template_CosmonautsT1.nii.gz')
#     ffFIX = ffATL_Astro_new
#     ffMOV = ffATLold
#     reg   = run_reg(ffFIX,ffMOV)
#     print('Registered T_template_AstronautsT1.nii.gz!')
#     ffFIX = ffATL_Cosmo_new
#     ffMOV = ffATLold
#     reg   = run_reg(ffFIX,ffMOV)
#     print('Registered T_template_CosmonautsT1.nii.gz!')
#     # now, do some manual fixing to fiducials and eyeball/lens-segmentations...
#     # ... done. Results in D:\Dropbox\Projects\AstronautT1\data\finalTemplatesJournal_2019
    
#     #%% nipype doesn't work on Windows... fix!
#     #from nipype.interfaces.base import CommandLine
#     #CommandLine("dir").run()
#     from nipype.interfaces.base import CommandLine
#     CommandLine("dir").run()

    
#     #%%%
#     ffATL = r'D:\GeTang\demo\volATL01.nii.gz'
#     ffSUB = r'D:\GeTang\demo\volSUB02.nii'
#     prefixOUT = r'D:\GeTang\demo\T_ATL_to_SUB_'
#     ffOUT = r'D:\GeTang\demo\T_ATL_to_SUB_volATLDeformed.nii.gz'
#     reg = regAnts(ffSUB,ffATL, 
#                   output_prefix = prefixOUT, 
#                   output_warped_image=ffOUT,
#                   transform_parameters=[(0.5,), (0.25, 3.0, 0.0)],
#                   smoothing_sigmas = [[2,1], [4,2,1]],
#                   shrink_factors = [[4,2], [8,4,2]],
#                   number_of_iterations=[[500, 250], [100, 70, 50]],
#                   composite_trf = True,
#                   convergence_threshold = [1.e-7, 1.e-8],
#                   verbose = True,
#                   initial_geometric_Align=True)
    
#     # subprocess.check_output(['wsl', 'echo', '$PATH'])
#     # subprocess.check_output(['wsl', 'export', 'PATH=${ANTSPATH}:$PATH'])
#     # subprocess.check_output(['wsl', 'ls'])
#     # subprocess.check_call(['wsl', 'ls'])
    
#     a = convert_cmd_to_linux(reg.cmdline)
#     b = 'wsl' + ' ' + a
#     print(b)
#     subprocess.check_output(b)
    
#     #%%
    
#     reg_0 = regAnts(ffSUB,ffATL, 
#                   output_prefix = prefixOUT, 
#                   output_warped_image=ffOUT,
#                   transform_parameters=[(2.0,), (0.25, 3.0, 0.0)],
#                   smoothing_sigmas = [[2,1], [3,2,1]],
#                   shrink_factors = [[4,2], [4,3,2]],
#                   number_of_iterations=[[1500, 200], [70, 90, 50]],
#                   convergence_threshold = [1.e-8, 1.e-9],
#                   verbose = True,
#                   initial_geometric_Align=True)
    
#     reg = regAnts(ffSUB,ffATL, 
#                   output_prefix = prefixOUT, 
#                   output_warped_image=ffOUT,
#                   transform_parameters=[(0.5,), (0.25, 3.0, 0.0)],
#                   smoothing_sigmas = [[2,1], [4,2,1]],
#                   shrink_factors = [[4,2], [8,4,2]],
#                   number_of_iterations=[[500, 250], [100, 70, 50]],
#                   convergence_threshold = [1.e-7, 1.e-8],
#                   verbose = True,
#                   initial_geometric_Align=True)
    
#     reg_2 = regAnts(ffSUB,ffATL, 
#                   output_prefix = prefixOUT, 
#                   output_warped_image=ffOUT,
#                   transform_parameters=[(0.5,), (0.25, 3.0, 0.0)],
#                   smoothing_sigmas = [[2,1], [4,2,1]],
#                   shrink_factors = [[4,2], [8,4,2]],
#                   number_of_iterations=[[500, 250], [70, 90, 20]],
#                   convergence_threshold = [1.e-6, 1.e-6],
#                   verbose = True,
#                   initial_geometric_Align=True)
    
#     reg_3 = regAnts(ffSUB,ffATL, 
#                   output_prefix = prefixOUT, 
#                   output_warped_image=ffOUT,
#                   transform_parameters=[(0.5,), (0.25, 3.0, 0.0)],
#                   smoothing_sigmas = [[2,1], [4,2,1]],
#                   shrink_factors = [[4,2], [8,4,2]],
#                   number_of_iterations=[[500, 250], [70, 90, 50]],
#                   convergence_threshold = [1.e-6, 1.e-6],
#                   verbose = True,
#                   initial_geometric_Align=True)
    
#     reg_4 = regAnts(ffSUB,ffATL, 
#                   output_prefix = prefixOUT, 
#                   output_warped_image=ffOUT,
#                   transform_parameters=[(0.5,), (0.25, 3.0, 0.0)],
#                   smoothing_sigmas = [[2,1], [2,1,0]],
#                   shrink_factors = [[4,2], [4,2,1]],
#                   number_of_iterations=[[500, 250], [180, 90, 20]],
#                   convergence_threshold = [1.e-6, 1.e-7],
#                   verbose = True,
#                   initial_geometric_Align=True)
    
#     reg = regAnts(ffSUB,ffATL, 
#                   output_prefix = prefixOUT, 
#                   output_warped_image=ffOUT,
#                   transform_parameters=[(0.5,), (0.25, 3.0, 0.0)],
#                   smoothing_sigmas = [[1], [2,1,0]],
#                   shrink_factors = [[4], [4,2,1]],
#                   number_of_iterations=[[250], [180, 90, 20]],
#                   convergence_threshold = [1.e-6, 1.e-7],
#                   verbose = True,
#                   initial_geometric_Align=True)
    
#     #import subprocess
#     #subprocess.run(reg.cmdline)
#     subprocess.check_output(['wsl', 'ls'])
#     subprocess.check_call(['wsl', 'ls'])
#     print('\nConverted:\n')
#     a = convert_cmd_to_linux(reg.cmdline)
#     subprocess.check_call(['antsRegistration', '--version'])
#     print(convert_cmd_to_linux(reg.cmdline))
    

# '''
#         ,output_prefix='output_',
#             transforms=['Affine', 'SyN'],
#             transform_parameters=[(2.0,), (0.25, 3.0, 0.0)],
#             number_of_iterations=[[1500, 200], [70, 90, 20]],
#             metric=['Mattes','CC'],
#             metric_weight=[1,1],
#             num_threads = 4,
#             radius_or_number_of_bins = [32,2],
#             sampling_strategy = ['Random', None],
#             sampling_percentage = [0.05, None],
#             convergence_threshold = [1.e-8, 1.e-9],
#             convergence_window_size = [20,20],
#             smoothing_sigmas = [[2,1], [3,2,1]],
#             sigma_units = ['vox','vox'],
#             shrink_factors = [[4,2], [4,3,2]],
#             use_histogram_matching = [True, True],
#             output_warped_image = 'output_warped_image.nii.gz' )
# '''

# '''
# example commandline:
    



# '''
