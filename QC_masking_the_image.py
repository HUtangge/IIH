"""
Ge Tang : After Getting the mask on the template, then register it to the individual images, and mask the relative area
1. Load the mask, the transforms
2. resample using the resample scalar/vector/DWI volume module
3. get the masked area
4. calculate the QC
"""
#%% Resample the mask
original_image_node = slicer.util.getNode('sub-18_ses-01_T1w_conformed_corrected_harmonized')
mask_image_node = slicer.util.getNode('Mask_R_eyeball_IIH02mmT1')
resampled_mask_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
parameters = {
    "inputVolume": mask_image_node.GetID(),
    "referenceVolume": original_image_node.GetID(),
    "outputVolume": resampled_mask_node.GetID(),
    "interpolationMode": "linear"
}
slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, parameters)

# Turn the scalar volume to float in order to do simple muptiplication
resampled_mask_node = slicer.util.getNode('Volume')
resampled_mask_float_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
parameters = {
    "InputVolume": resampled_mask_node.GetID(),
    "OutputVolume": resampled_mask_float_node.GetID(),
    "Type": "Float"
}
slicer.cli.runSync(slicer.modules.castscalarvolume, None, parameters)

# Multiply the two images to get the masked image
import SimpleITK as sitk
import sitkUtils

def convertVolumeNodeToSITKImage(volumeNode):
    # Get the Slicer volume as a SimpleITK image
    sitkImage = sitkUtils.PullVolumeFromSlicer(volumeNode)
    return sitkImage

inputvolumenode1 = slicer.util.getNode('sub-18_ses-01_T1w_conformed_corrected_harmonized')
inputimage1 = convertVolumeNodeToSITKImage(inputvolumenode1)
inputvolumenode2 = slicer.util.getNode('Volume_3')
inputimage2 = convertVolumeNodeToSITKImage(inputvolumenode2)

# Access the filter
multiplyFilter = sitk.MultiplyImageFilter()

# set the number of thread for computation
multiplyFilter.SetNumberOfThreads(64)

# Execute the filter
outputImage = multiplyFilter.Execute(inputimage1, inputimage2)
sitk.WriteImage(outputImage, "D:/users/getang/IIH/tmp/Multiplied_Volume.nii.gz")

outputVolumeNode = sitkUtils.PushVolumeToSlicer(outputImage, className="vtkMRMLScalarVolumeNode")
outputVolumeNode.SetName("maksed_image")

#%%
"""
Ge Tang : 
Calculating the QCs
"""
import nibabel as nb
import numpy as np

import scipy.ndimage as nd
from math import sqrt
from scipy.stats import kurtosis 

# Functions
def efc_masked(img, framemask=None):
    """
    This is adjusted to the masked image to calculate the efc within masked area
    the framemask is the mask for the boundary

    """

    if framemask is None:
        framemask = np.zeros_like(img, dtype=np.uint8)

    n_vox = np.sum(framemask)
    print(n_vox)
    # Calculate the maximum value of the EFC (which occurs any time all
    # voxels have the same value)
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * np.log(1.0 / np.sqrt(n_vox))
    print(efc_max)

    # Calculate the total image energy
    b_max = np.sqrt((img[framemask == 1] ** 2).sum())
    print(b_max)

    # Calculate EFC (add 1e-16 to the image data to keep log happy)
    return float(
        (1.0 / efc_max)
        * np.sum((img[framemask == 1] / b_max) * np.log((img[framemask == 1] + 1e-16) / b_max))
    )

def efc(img, framemask=None):
    """
    Calculate the :abbr:`EFC (Entropy Focus Criterion)` [Atkinson1997]_.
    Uses the Shannon entropy of voxel intensities as an indication of ghosting
    and blurring induced by head motion. A range of low values is better,
    with EFC = 0 for all the energy concentrated in one pixel.

    .. math::

        \text{E} = - \sum_{j=1}^N \frac{x_j}{x_\text{max}}
        \ln \left[\frac{x_j}{x_\text{max}}\right]

    with :math:`x_\text{max} = \sqrt{\sum_{j=1}^N x^2_j}`.

    The original equation is normalized by the maximum entropy, so that the
    :abbr:`EFC (Entropy Focus Criterion)` can be compared across images with
    different dimensions:

    .. math::

        \text{EFC} = \left( \frac{N}{\sqrt{N}} \, \log{\sqrt{N}^{-1}} \right) \text{E}

    :param numpy.ndarray img: input data
    :param numpy.ndarray framemask: a mask of empty voxels inserted after a rotation of
      data

    """
    if framemask is None:
        framemask = np.zeros_like(img, dtype=np.uint8)

    n_vox = np.sum(1 - framemask)
    print(n_vox)
    # Calculate the maximum value of the EFC (which occurs any time all
    # voxels have the same value)
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * np.log(1.0 / np.sqrt(n_vox))

    # Calculate the total image energy
    b_max = np.sqrt((img[framemask == 0] ** 2).sum())
    print(b_max)

    # Calculate EFC (add 1e-16 to the image data to keep log happy)
    return float(
        (1.0 / efc_max)
        * np.sum((img[framemask == 0] / b_max) * np.log((img[framemask == 0] + 1e-16) / b_max))
    )

def summary_stats(data, pvms, airmask=None, erode=True):
    r"""
    Estimates the mean, the median, the standard deviation,
    the kurtosis,the median absolute deviation (mad), the 95\%
    and the 5\% percentiles and the number of voxels (summary\_\*\_n)
    of each tissue distribution.

    .. warning ::

        Sometimes (with datasets that have been partially processed), the air
        mask will be empty. In those cases, the background stats will be zero
        for the mean, median, percentiles and kurtosis, the sum of voxels in
        the other remaining labels for ``n``, and finally the MAD and the
        :math:`\sigma` will be calculated as:

        .. math ::

            \sigma_\text{BG} = \sqrt{\sum \sigma_\text{i}^2}


    """
    from statsmodels.stats.weightstats import DescrStatsW
    from statsmodels.robust.scale import mad

    output = {}
    for label, probmap in pvms.items():
        wstats = DescrStatsW(data=data.reshape(-1), weights=probmap.reshape(-1))
        nvox = probmap.sum()
        p05, median, p95 = wstats.quantile(
            np.array([0.05, 0.50, 0.95]),
            return_pandas=False,
        )
        thresholded = data[probmap > (0.5 * probmap.max())]

        output[label] = {
            "mean": float(wstats.mean),
            "median": float(median),
            "p95": float(p95),
            "p05": float(p05),
            "k": float(kurtosis(thresholded)),
            "stdv": float(wstats.std),
            "mad": float(mad(thresholded, center=median)),
            "n": float(nvox),
        }

    return output

def summary_stats_15(img, pvms, airmask=None, erode=True):
    from statsmodels.robust.scale import mad
    FSL_FAST_LABELS = {'csf': 1, 'gm': 2, 'wm': 3, 'bg': 0}

    # Check type of input masks
    dims = np.squeeze(np.array(pvms)).ndim
    print(dims)
    if dims == 4:
        # If pvms is from FSL FAST, create the bg mask
        stats_pvms = [np.zeros_like(img)] + pvms
    elif dims == 3:
        stats_pvms = [np.ones_like(pvms) - pvms, pvms]
    else:
        raise RuntimeError('Incorrect image dimensions ({0:d})'.format(
            np.array(pvms).ndim))

    if airmask is not None:
        stats_pvms[0] = airmask

    labels = list(FSL_FAST_LABELS.items())
    if len(stats_pvms) == 2:
        labels = list(zip(['bg', 'fg'], list(range(2))))

    output = {}
    for k, lid in labels:
        print(k)
        print(lid)
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[stats_pvms[lid] > 0.85] = 1

        if erode:
            struc = nd.generate_binary_structure(3, 2)
            mask = nd.binary_erosion(
                mask, structure=struc).astype(np.uint8)

        nvox = float(mask.sum())

        output[k] = {
            'mean': float(img[mask == 1].mean()),
            'stdv': float(img[mask == 1].std()),
            'median': float(np.median(img[mask == 1])),
            'mad': float(mad(img[mask == 1])),
            'p95': float(np.percentile(img[mask == 1], 95)),
            'p05': float(np.percentile(img[mask == 1], 5)),
            'k': float(kurtosis(img[mask == 1])),
            'n': nvox,
        }

    if 'bg' not in output:
        output['bg'] = {
            'mean': 0.,
            'median': 0.,
            'p95': 0.,
            'p05': 0.,
            'k': 0.,
            'stdv': sqrt(sum(val['stdv']**2
                             for _, val in list(output.items()))),
            'mad': sqrt(sum(val['mad']**2
                            for _, val in list(output.items()))),
            'n': sum(val['n'] for _, val in list(output.items()))
        }

    if 'bg' in output and output['bg']['mad'] == 0.0 and output['bg']['stdv'] > 1.0:
        output['bg']['mad'] = output['bg']['stdv'] / DIETRICH_FACTOR
    return output

def summary_stats_eye_15(img, pvms, airmask=None, erode=True):
    from statsmodels.robust.scale import mad
    FSL_FAST_LABELS = {'eyeball': 1, 'eyeball_sheath': 2}

    # Check type of input masks
    dims = np.squeeze(np.array(pvms)).ndim
    print(dims)
    if dims == 4:
        # If pvms is from FSL FAST, create the bg mask
        stats_pvms = [np.zeros_like(img)] + pvms
    elif dims == 3:
        stats_pvms = [np.ones_like(pvms) - pvms, pvms]
    else:
        raise RuntimeError('Incorrect image dimensions ({0:d})'.format(
            np.array(pvms).ndim))

    if airmask is not None:
        stats_pvms[0] = airmask

    labels = list(FSL_FAST_LABELS.items())

    output = {}
    for k, lid in labels:
        print(k)
        print(lid)
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[stats_pvms[lid] > 0.85] = 1

        if erode:
            struc = nd.generate_binary_structure(3, 2)
            mask = nd.binary_erosion(
                mask, structure=struc).astype(np.uint8)

        nvox = float(mask.sum())

        output[k] = {
            'mean': float(img[mask == 1].mean()),
            'stdv': float(img[mask == 1].std()),
            'median': float(np.median(img[mask == 1])),
            'mad': float(mad(img[mask == 1])),
            'p95': float(np.percentile(img[mask == 1], 95)),
            'p05': float(np.percentile(img[mask == 1], 5)),
            'k': float(kurtosis(img[mask == 1])),
            'n': nvox,
        }

    if 'bg' not in output:
        output['bg'] = {
            'mean': 0.,
            'median': 0.,
            'p95': 0.,
            'p05': 0.,
            'k': 0.,
            'stdv': sqrt(sum(val['stdv']**2
                             for _, val in list(output.items()))),
            'mad': sqrt(sum(val['mad']**2
                            for _, val in list(output.items()))),
            'n': sum(val['n'] for _, val in list(output.items()))
        }

    if 'bg' in output and output['bg']['mad'] == 0.0 and output['bg']['stdv'] > 1.0:
        output['bg']['mad'] = output['bg']['stdv'] / DIETRICH_FACTOR
    return output

def snr(mu_fg, sigma_fg, n):
    r"""
    Calculate the :abbr:`SNR (Signal-to-Noise Ratio)`.
    The estimation may be provided with only one foreground region in
    which the noise is computed as follows:

    .. math::

        \text{SNR} = \frac{\mu_F}{\sigma_F\sqrt{n/(n-1)}},

    where :math:`\mu_F` is the mean intensity of the foreground and
    :math:`\sigma_F` is the standard deviation of the same region.

    :param float mu_fg: mean of foreground.
    :param float sigma_fg: standard deviation of foreground.
    :param int n: number of voxels in foreground mask.

    :return: the computed SNR

    """
    return float(mu_fg / (sigma_fg * sqrt(n / (n - 1))))


#%%
maskedarray = slicer.util.arrayFromVolume(outputVolumeNode)
result_efc = efc_masked(maskedarray)

#%% 
"""
Ge Tang:
Steps for getting the regional QC
!!!The inudata in the MRIQC 15.1 is in the folder ComputeIQMs\...\harmonize\sub-18_ses-01_T1w_conformed_corrected_harmonized.nii.gz
!!!The probability map is processed using the raw image 
!!!The airmask is in the folder \AirMaskWorkflow\...\ArtifactMask\sub-18_ses-01_T1w_conformed_hat.nii.gz

1. use fsl fast to get the pvms
bet sub-18_ses-01_T1w.nii.gz out_sub-18_ses-01_T1w.nii.gz
fast -o output_basename -v -t 1 -g -S 1 out_sub-18_ses-01_T1w.nii.gz
fast -t 1 -o segment -g -S 1 out_sub-18_ses-01_T1w.nii.gz
2. run the following code
"""

fname0 = "/Users/getang/Documents/EarthResearch/IIH/tmp/segment_eye_pve_0.nii.gz"
fname1= "/Users/getang/Documents/EarthResearch/IIH/tmp/segment_eye_pve_1.nii.gz"
fname2= "/Users/getang/Documents/EarthResearch/IIH/tmp/segment_eye_pve_2.nii.gz"

pvms = {label: nb.load(fname).get_fdata()
        for label, fname in zip(("csf", "gm", "wm"), (fname0, fname1, fname2))
        }

pvms = {label: nb.load(fname).get_fdata()
        for label, fname in zip(("csf", "gm"), (fname0, fname1))
        }

inudata = nb.load("/Users/getang/Documents/EarthResearch/IIH/tmp/Multiplied_Volume.nii.gz")
inudata = np.nan_to_num(inudata.get_fdata())
inudata[inudata < 0] = 0

# Calculate the efc
mask = nb.load("/Users/getang/Documents/EarthResearch/IIH/tmp/Mask.nii.gz")
mask = np.nan_to_num(mask.get_fdata())

result_efc = efc_masked(inudata, mask)
print(f'the efc is {result_efc}')


#%%
# Calculate the snr
stats = summary_stats_15(inudata, pvms)

snrvals = []
results = {}
for tlabel in ("csf", "wm", "gm"):
    snrvals.append(
        snr(
            stats[tlabel]["median"],
            stats[tlabel]["stdv"],
            stats[tlabel]["n"],
        )
    )
    print(snrvals)
    results[tlabel] = snrvals[-1]
results["total"] = float(np.mean(snrvals))
print(f'the snrs are {results}')

#%%
for label, fname in zip(("csf", "gm", "wm"), ("a", "b", "b")):
    print(label)
    print(fname)
# %% Testing code
"""
Tang Ge: The following is the code for testing before integrated to the script
"""
# This is for the files from the mriqc 
# I can load the different segmentation probability map separately into the 
import numpy as np
import pickle
pvms = np.load("/Users/getang/Documents/EarthResearch/IIH/tmp/pvms_sub-18_ses-01_T1w.npy", allow_pickle=True)
pvms = pvms[()]

#%%
results = {}

fname0 = "/Users/getang/Documents/EarthResearch/IIH/tmp/segment_pve_0.nii.gz"
fname1= "/Users/getang/Documents/EarthResearch/IIH/tmp/segment_pve_1.nii.gz"
fname2= "/Users/getang/Documents/EarthResearch/IIH/tmp/segment_pve_2.nii.gz"

pvmdata = []
for fname in [fname0, fname1, fname2]:
    print(fname)
    pvmdata.append(nb.load(fname).get_fdata().astype(np.float32))

inudata = nb.load("/Users/getang/Documents/EarthResearch/IIH/tmp/sub-18_ses-01_T1w_conformed_corrected_harmonized.nii.gz")
inudata = np.nan_to_num(inudata.get_fdata())
inudata[inudata < 0] = 0

airmask = nb.load("/Users/getang/Documents/EarthResearch/IIH/tmp/sub-18_ses-01_T1w_conformed_air.nii.gz")
airmask = airmask.get_fdata()

results['efc_wholebrain'] = efc(inudata)

# For the function, I am using the one from the MRIQC version 15.1
stats = summary_stats_15(inudata, pvmdata, airmask)

snrvals = []
for tlabel in ("csf", "wm", "gm"):
    snrvals.append(
        snr(
            stats[tlabel]["median"],
            stats[tlabel]["stdv"],
            stats[tlabel]["n"],
        )
    )
    print(snrvals)
    results['snr_' + tlabel] = snrvals[-1]
results["snr_total_wholebrain"] = float(np.mean(snrvals))

# This is for the eye segmentation
fname0 = "/Users/getang/Documents/EarthResearch/IIH/tmp/segment_eye_pve_0.nii.gz"
fname1= "/Users/getang/Documents/EarthResearch/IIH/tmp/segment_eye_pve_1.nii.gz"

pvmdata = []
for fname in [fname0, fname1]:
    print(fname)
    pvmdata.append(nb.load(fname).get_fdata().astype(np.float32))

inudata = nb.load("/Users/getang/Documents/EarthResearch/IIH/tmp/Multiplied_Volume.nii.gz")
inudata = np.nan_to_num(inudata.get_fdata())
inudata[inudata < 0] = 0

# Calculate the efc
mask = nb.load("/Users/getang/Documents/EarthResearch/IIH/tmp/Mask.nii.gz")
mask = np.nan_to_num(mask.get_fdata())

results['efc_eyeball_front'] = efc_masked(inudata, mask)

stats = summary_stats_eye_15(inudata, pvmdata)

snrvals = []
results = {}
for tlabel in ("eyeball", "eyeball_sheath"):
    snrvals.append(
        snr(
            stats[tlabel]["median"],
            stats[tlabel]["stdv"],
            stats[tlabel]["n"],
        )
    )
    print(snrvals)
    results['snr_' + tlabel] = snrvals[-1]
results["snr_total_eyeball_front"] = float(np.mean(snrvals))

# %%
from nipype.utils.filemanip import loadpkl
res = loadpkl('/Users/getang/Documents/EarthResearch/IIH/tmp/_inputs.pklz')
# %% This is for loading the transforms and get the masked images and the mask
# Ge Tang
#%% 
import sys
sys.path.append(r"D:\users\getang\SANS\Slicertools")
import file_search_tool as fs
import os
import re
import SimpleITK as sitk
import sitkUtils

# Functions
def convertVolumeNodeToSITKImage(volumeNode):
    # Get the Slicer volume as a SimpleITK image
    sitkImage = sitkUtils.PullVolumeFromSlicer(volumeNode)
    return sitkImage

project_path = r'D:\users\getang\IIH'
pnTrf = r'%s\data\Rawdata'%project_path
pnMask = r'%s\data\Masks'%project_path
fnMaskptn = r'Mask_%s_%s%s.nii.gz'
pnMRI = r'%s\IIH_MRIQC_15_1'%project_path
fnMRIRawptn = r'sub*[0-9]_%sw.nii.gz'

for iterc, modality in enumerate([['IIH02mm', 'T1', 'eyeball'], ['IIH02mm', 'T1', 'retroorbital'], ['IIH02mm', 'T2', 'eyeball'], ['IIH02mm', 'T2', 'retroorbital']]): 
    fnMask = fnMaskptn%(modality[2], modality[0], modality[1])
    fnMRIRaw = fnMRIRawptn%(modality[1])
    fl = fs.locateFiles(fnMRIRaw, pnMRI, level=4)
    
    # for each file
    for idx, ff in enumerate(fl):
        ff = ff.replace('/','\\')
        print(f'Processing {ff}')

        # extract fn root: sub-cosmonaut01_ses-postflight_T1w_n4
        pn, fn = os.path.split(ff)
        fnroot, fnext = fs.splitext(fn)        
        which_sub = re.search('sub-(.+?)_ses', fnroot).group(1)
        fnAFF = 'trf-Temp_to_%s_AFF-Denoised_%s.mat'%(which_sub, fnroot)
        fnDEF = 'trf-Temp_to_%s_DEF-Denoised_%s.nii.gz'%(which_sub, fnroot) 

        success,nMask = loadVolume(os.path.join(pnMask,fnMask),returnNode=True)
        success,nMRIRaw = loadVolume(ff,returnNode=True)        
        success,nMRIProcessed = loadVolume(os.path.join(pn, fnroot + '_conformed_corrected_harmonized.nii.gz'),returnNode=True)     

        # Set the name for the masked volume
        fnMaskMRI = 'Mask_%s_%s'%(modality[2], fn)
        fnMaskfloatMRI = 'Mask_float_%s_%s'%(modality[2], fn)
        fnMaskedMRIRaw = 'Masked_%s_%s'%(modality[2], fn)
        fnMaskedMRIProcessed = 'Masked_%s_%s_conformed_corrected_harmonized.nii.gz'%(modality[2], fnroot)

        if os.path.exists(os.path.join(pnTrf, fnroot[:-4], fnAFF)):
            print("The path exists.")
            success, nTrfATLtoMRI_AFF = slicer.util.loadTransform(os.path.join(pnTrf, fnroot[:-4],fnAFF), returnNode=True)
            success, nTrfATLtoMRI_DEF = slicer.util.loadTransform(os.path.join(pnTrf, fnroot[:-4],fnDEF), returnNode=True)        
            nTrfATLtoMRI_AFF.SetAndObserveTransformNodeID(nTrfATLtoMRI_DEF.GetID())
            nMask.SetAndObserveTransformNodeID(nTrfATLtoMRI_AFF.GetID())
            nMask.HardenTransform()

            resampled_nMask = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            resampled_nMask.SetName('Resampled_mask')
            parameters = {
                "inputVolume": nMask.GetID(),
                "referenceVolume": nMRIRaw.GetID(),
                "outputVolume": resampled_nMask.GetID(),
                "interpolationMode": "linear"
            }
            slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, parameters)
            saveNode(resampled_nMask, os.path.join(pn.replace('IIH_BIDS', 'IIH_MRIQC_15_1\\out'), fnMaskMRI))

            resampled_nMask_float = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            resampled_nMask_float.SetName('Resampled_mask_float')
            parameters = {
                "InputVolume": resampled_nMask.GetID(),
                "OutputVolume": resampled_nMask_float.GetID(),
                "Type": "Float"
            }            
            slicer.cli.runSync(slicer.modules.castscalarvolume, None, parameters)
            saveNode(resampled_nMask_float, os.path.join(pn.replace('IIH_BIDS', 'IIH_MRIQC_15_1\\out'), fnMaskfloatMRI))

            inputimage1 = convertVolumeNodeToSITKImage(resampled_nMask)
            inputimage2 = convertVolumeNodeToSITKImage(nMRIRaw)
            inputimage3 = convertVolumeNodeToSITKImage(resampled_nMask_float)
            inputimage4 = convertVolumeNodeToSITKImage(nMRIProcessed)
            # Set up the filter
            multiplyFilter = sitk.MultiplyImageFilter()
            multiplyFilter.SetNumberOfThreads(64)
            outputImageRaw = multiplyFilter.Execute(inputimage1, inputimage2)
            sitk.WriteImage(outputImageRaw, os.path.join(pn.replace('IIH_BIDS', 'IIH_MRIQC_15_1\\out'),fnMaskedMRIRaw))
            outputImageProcessed = multiplyFilter.Execute(inputimage3, inputimage4)
            sitk.WriteImage(outputImageProcessed, os.path.join(pn.replace('IIH_BIDS', 'IIH_MRIQC_15_1\\out'),fnMaskedMRIProcessed))
        else:
            print("The path does not exist.")

        su.closeScene()       

#%%
"""
Ge Tang: This is for creating the cut plane for the segmentation, 
I use the orbital rim plane as the front part of the retroorbital segmentation,
and the (x,y,z), (x,y-43,z) as the end plane 
"""
#%% Set up the plane for segmentation
import numpy as np
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

def getOrCreateMarkupsPlaneNode(namePlaneNode, centerpoint, planenormal):
    """

    Parameters
    ----------
    namePlaneNode : TYPE
        DESCRIPTION.
    centerpoint : TYPE
        a 3 by 1 numpy array
    planenormal : TYPE
        a 3 by 1 numpy array

    Returns
    -------
    planeNode : TYPE
        vtkMRMLMarkupsPlaneNode

    """
    planeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", namePlaneNode)
    planeNode.SetCenter(centerpoint)
    planeNode.SetNormal(planenormal)
    visFid_SetVisibility(planeNode.GetName(), visibility=0.4, locked=1, textScale=0, color = (255,255,0),
                         glyph_scale = 0)
    return planeNode

ptsOrb = su.arrayFromFiducialList('fids_IIH02mmT1', listFidNames=['orbital_rim_L_lat', 'orbital_rim_L_med', 'orbital_rim_L_sup', 'orbital_rim_L_inf'])
normal,offset,R = lstsqPlaneEstimation(ptsOrb)
plane = getOrCreateMarkupsPlaneNode('plane_L', offset, normal)

reformat_logic = slicer.vtkSlicerReformatLogic()
layout_manager = slicer.app.layoutManager()
slice_node = layout_manager.sliceWidget('Red').mrmlSliceNode()
reformat_logic.SetSliceNormal(slice_node, normal[0], normal[1], normal[2])

# 02mm_T2 image
offset_L_1=np.array([-31.64853025, 91.10237602, -17.63189937])
offset_L_2=np.array([-31.64853025, 48.10237602, -17.63189937])

offset_R_1=np.array([32.69854682, 90.32977069, -18.22689294])
offset_R_2=np.array([32.69854682, 47.32977069, -18.22689294])

# 02mm_T1 image
offset_L_1=np.array([-31.68119344, 90.89813936, -19.21019086])
offset_L_2=np.array([-31.68119344, 47.89813936, -19.21019086])

offset_R_1=np.array([32.548159, 90.72925756, -19.7715574])
offset_R_2=np.array([32.548159, 47.72925756, -19.7715574])

#%%
#%%
# Ge Tang: This is running in the python
import sys
sys.path.append(r"D:\users\getang\SANS\Slicertools")
import file_search_tool as fs
import os

import nibabel as nb
import numpy as np
import pandas as pd
import scipy.ndimage as nd
from math import sqrt
from scipy.stats import kurtosis 

#%% Functions 
def efc_masked(img, framemask=None):
    """
    This is adjusted to the masked image to calculate the efc within masked area
    the framemask is the mask for the boundary

    """

    if framemask is None:
        framemask = np.zeros_like(img, dtype=np.uint8)

    n_vox = np.sum(framemask)
    print(n_vox)
    # Calculate the maximum value of the EFC (which occurs any time all
    # voxels have the same value)
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * np.log(1.0 / np.sqrt(n_vox))
    print(efc_max)

    # Calculate the total image energy
    b_max = np.sqrt((img[framemask == 1] ** 2).sum())
    print(b_max)

    # Calculate EFC (add 1e-16 to the image data to keep log happy)
    return float(
        (1.0 / efc_max)
        * np.sum((img[framemask == 1] / b_max) * np.log((img[framemask == 1] + 1e-16) / b_max))
    )

def efc(img, framemask=None):
    """
    Calculate the :abbr:`EFC (Entropy Focus Criterion)` [Atkinson1997]_.
    Uses the Shannon entropy of voxel intensities as an indication of ghosting
    and blurring induced by head motion. A range of low values is better,
    with EFC = 0 for all the energy concentrated in one pixel.

    .. math::

        \text{E} = - \sum_{j=1}^N \frac{x_j}{x_\text{max}}
        \ln \left[\frac{x_j}{x_\text{max}}\right]

    with :math:`x_\text{max} = \sqrt{\sum_{j=1}^N x^2_j}`.

    The original equation is normalized by the maximum entropy, so that the
    :abbr:`EFC (Entropy Focus Criterion)` can be compared across images with
    different dimensions:

    .. math::

        \text{EFC} = \left( \frac{N}{\sqrt{N}} \, \log{\sqrt{N}^{-1}} \right) \text{E}

    :param numpy.ndarray img: input data
    :param numpy.ndarray framemask: a mask of empty voxels inserted after a rotation of
      data

    """
    if framemask is None:
        framemask = np.zeros_like(img, dtype=np.uint8)

    n_vox = np.sum(1 - framemask)
    print(n_vox)
    # Calculate the maximum value of the EFC (which occurs any time all
    # voxels have the same value)
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * np.log(1.0 / np.sqrt(n_vox))

    # Calculate the total image energy
    b_max = np.sqrt((img[framemask == 0] ** 2).sum())
    print(b_max)

    # Calculate EFC (add 1e-16 to the image data to keep log happy)
    return float(
        (1.0 / efc_max)
        * np.sum((img[framemask == 0] / b_max) * np.log((img[framemask == 0] + 1e-16) / b_max))
    )

def summary_stats_eye_15(img, pvms, FSL_FAST_LABELS, airmask=None, erode=True):
    from statsmodels.robust.scale import mad
    DIETRICH_FACTOR = 1.0 / sqrt(2 / (4 - np.pi))

    # Check type of input masks
    dims = np.squeeze(np.array(pvms)).ndim
    print(dims)
    if dims == 4:
        # If pvms is from FSL FAST, create the bg mask
        stats_pvms = [np.zeros_like(img)] + pvms
    elif dims == 3:
        stats_pvms = [np.ones_like(pvms) - pvms, pvms]
    else:
        raise RuntimeError('Incorrect image dimensions ({0:d})'.format(
            np.array(pvms).ndim))

    if airmask is not None:
        stats_pvms[0] = airmask

    labels = list(FSL_FAST_LABELS.items())

    output = {}
    for k, lid in labels:
        print(k)
        print(lid)
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[stats_pvms[lid] > 0.85] = 1

        if erode:
            struc = nd.generate_binary_structure(3, 2)
            mask = nd.binary_erosion(
                mask, structure=struc).astype(np.uint8)

        nvox = float(mask.sum())

        output[k] = {
            'mean': float(img[mask == 1].mean()),
            'stdv': float(img[mask == 1].std()),
            'median': float(np.median(img[mask == 1])),
            'mad': float(mad(img[mask == 1])),
            'p95': float(np.percentile(img[mask == 1], 95)),
            'p05': float(np.percentile(img[mask == 1], 5)),
            'k': float(kurtosis(img[mask == 1])),
            'n': nvox,
        }

    if 'bg' not in output:
        output['bg'] = {
            'mean': 0.,
            'median': 0.,
            'p95': 0.,
            'p05': 0.,
            'k': 0.,
            'stdv': sqrt(sum(val['stdv']**2
                             for _, val in list(output.items()))),
            'mad': sqrt(sum(val['mad']**2
                            for _, val in list(output.items()))),
            'n': sum(val['n'] for _, val in list(output.items()))
        }

    if 'bg' in output and output['bg']['mad'] == 0.0 and output['bg']['stdv'] > 1.0:
        output['bg']['mad'] = output['bg']['stdv'] / DIETRICH_FACTOR
    return output

def snr(mu_fg, sigma_fg, n):
    r"""
    Calculate the :abbr:`SNR (Signal-to-Noise Ratio)`.
    The estimation may be provided with only one foreground region in
    which the noise is computed as follows:

    .. math::

        \text{SNR} = \frac{\mu_F}{\sigma_F\sqrt{n/(n-1)}},

    where :math:`\mu_F` is the mean intensity of the foreground and
    :math:`\sigma_F` is the standard deviation of the same region.

    :param float mu_fg: mean of foreground.
    :param float sigma_fg: standard deviation of foreground.
    :param int n: number of voxels in foreground mask.

    :return: the computed SNR

    """
    return float(mu_fg / (sigma_fg * sqrt(n / (n - 1))))

#%%
project_path = r'D:\users\getang\IIH\IIH_MRIQC_15_1\out'
fninuptn = r'sub*[0-9]_%sw_conformed_corrected_harmonized.nii.gz'
fneyeinuptn = r'Masked_%s_sub*[0-9]_%sw_conformed_corrected_harmonized.nii.gz'
fnmaskptn = r'Mask_%s*[0-9]_%sw.nii.gz'
fnrotmaskptn = r'sub*[0-9]_%sw_conformed_rotmask.nii.gz'
fnairmaskptn = r'sub*[0-9]_%sw_conformed_air.nii.gz'
fnsegmentptn = r'segment_%s_%sw_pve_%s.nii.gz'

result = {}
results = pd.DataFrame()
for iterc, modality in enumerate([['IIH02mm', 'T1', 'eyeball']]): 
    fnmask = fs.locateFiles(fnmaskptn%(modality[2], modality[1]), project_path, level=3)
    fninu = fs.locateFiles(fninuptn%modality[1], project_path, level=3)
    fneyeinu = fs.locateFiles(fneyeinuptn%(modality[2], modality[1]), project_path, level=3)
    fnrotmask = fs.locateFiles(fnrotmaskptn%modality[1], project_path, level=3)
    fnairmask = fs.locateFiles(fnairmaskptn%modality[1], project_path, level=3)
    
    # for each file
    for idx, ff in enumerate(fnmask):
        idx = 0
        ff = fnmask[idx]
        pn, fn = os.path.split(ff)
        fnroot, fnext = fs.splitext(fn)        

        inudata = nb.load([s for s in fninu if fnroot.replace('Mask_%s_'%modality[2], '') in s][0])
        inudata = np.nan_to_num(inudata.get_fdata())
        inudata[inudata < 0] = 0
        rotdata = nb.load([s for s in fnrotmask if fnroot.replace('Mask_%s_'%modality[2], '') in s][0]).get_fdata().astype(np.uint8)
        result['efc_wholebrain'] = efc(inudata, rotdata)

        mask = nb.load(ff)
        mask = np.nan_to_num(mask.get_fdata())
        result['efc_%s'%modality[2]] = efc_masked(inudata, mask)

        fname0 = os.path.join(pn, fnsegmentptn%(modality[2], modality[1], '0'))
        fname1 = os.path.join(pn, fnsegmentptn%(modality[2], modality[1], '1'))
        pvmdata = []
        for fname in [fname0, fname1]:
            print(fname)
            pvmdata.append(nb.load(fname).get_fdata().astype(np.float32))

        eye_inudata = nb.load([s for s in fneyeinu if fnroot.replace('Mask_%s_'%modality[2], '') in s][0])
        eye_inudata = np.nan_to_num(eye_inudata.get_fdata())
        eye_inudata[eye_inudata < 0] = 0
        if modality[2] == 'eyeball':
            FSL_FAST_LABELS = {'fluid': 1, 'eyeballsheath': 2}
        elif modality[2] == 'retroorbital':
            FSL_FAST_LABELS = {'fat': 1, 'muscle': 2}
        
        stats = summary_stats_eye_15(eye_inudata, pvmdata, FSL_FAST_LABELS, erode=False)
        snrvals = []
        for tlabel in list(stats.keys()):
            snrvals.append(
                snr(
                    stats[tlabel]["median"],
                    stats[tlabel]["stdv"],
                    stats[tlabel]["n"],
                )
            )
            print(snrvals)
            if tlabel != 'bg':
                result['snr_' + tlabel] = snrvals[-1]
        result["snr_%s"%modality[2]] = float(np.mean(snrvals))

        result_df = pd.DataFrame(result, index = [fnroot.replace('Mask_%s_'%modality[2], '')])
        results = pd.concat([results, result_df], axis = 0)

df = results.groupby(level=0).agg(lambda x: x.dropna().iloc[0])

#%%
import nibabel as nib
nifti_image = nib.Nifti1Image(mask, affine=np.eye(4))
nib.save(nifti_image, 'D:/users/getang/IIH/tmp/mask_before_erode.nii.gz')
