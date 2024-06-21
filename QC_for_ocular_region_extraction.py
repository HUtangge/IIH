"""
Ge Tang: This is a file for getting the QC parameters from the 
"""
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