# Downsample images
#%%
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# Load the image
filename = r'/Users/getang/Documents/EarthResearch/IIH/tmp/sub-WISNEFO20180808A_T1w.nii'
newimagename = r'/Users/getang/Documents/EarthResearch/IIH/tmp/downsampled_sub-WISNEFO20180808A_T1w.nii'
image = nib.load(filename)

# Get the image data and affine matrix
data = image.get_fdata()
affine = image.affine

# Set the downsample factor
downsample_factor = 3

#%% Downsample the image data
new_data = zoom(data, (1 / downsample_factor, 1 / downsample_factor, 1 / downsample_factor))

# Update the affine matrix to account for the downsample
new_affine = affine.copy()
new_affine[:3, :3] *= downsample_factor

# Create a new image with the downsampled data and updated affine matrix
new_image = nib.Nifti1Image(new_data, new_affine)

#%% Save the downsampled image
nib.save(new_image, newimagename)


# %%

volumeNode = slicer.util.getNode('sub-WISNEFO20180808A_T1w') 
downsampledVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
desiredSpacing = '1,1,1'  # replace with your desired spacing

# Resample the image
parameters = {}
parameters["outputPixelSpacing"] = desiredSpacing
parameters["interpolationType"] = 'bspline'
parameters["InputVolume"] = volumeNode
parameters["OutputVolume"] = downsampledVolumeNode
parameters["outputPixelSpacing"] = desiredSpacing

slicer.cli.run(slicer.modules.resamplescalarvolume, None, parameters, wait_for_completion=True)

# Save the downsampled image
downsampledImagePath = 'path/to/your/downsampled_image.nii'  # replace with your desired file path
slicer.util.saveNode(downsampledVolumeNode, downsampledImagePath)


cliModule = slicer.modules.resamplescalarvolume
n=cliModule.cliModuleLogic().CreateNode()
for groupIndex in range(n.GetNumberOfParameterGroups()):
  print(f'Group: {n.GetParameterGroupLabel(groupIndex)}')
  for parameterIndex in range(n.GetNumberOfParametersInGroup(groupIndex)):
    print('  {0} [{1}]: {2}'.format(n.GetParameterName(groupIndex, parameterIndex),
      n.GetParameterTag(groupIndex, parameterIndex),n.GetParameterLabel(groupIndex, parameterIndex)))
