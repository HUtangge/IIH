# This file is for testing the ANTs 

# import pydicom

# filename = r'/Users/getang/Documents/EarthResearch/IIH/test/TemplateBuildingExample/BrainSlices/OASIS-TRT-20-10Slice121.nii.gz'
# ds = pydicom.dcmread(filename, force=True)

#%%
import nibabel as nib
import random
import os

def modify_voxels_nii(nii_path, voxel_coordinates, new_value):
    """
    Modify the value of specific voxels in a NIfTI file.
    
    :param nii_path: Path to the NIfTI file (.nii or .nii.gz).
    :param voxel_coordinates: List of 3D coordinates (i, j, k) of voxels to be changed.
    :param new_value: The value to which the voxel values should be changed.
    :return: Modified NIfTI image.
    """
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()

    # Modify the voxel values
    count = -1
    print(voxel_coordinates)
    for i, j in voxel_coordinates:
        count += 1
        print(i,j)
        data[i][j] = new_value[count]

    print(count)
    # Create a new NIfTI image with the modified data
    new_img = nib.Nifti1Image(data, nii_img.affine, nii_img.header)

    return new_img

#%%
voxels = [[(88, 189), (88, 190), (88, 191), (89, 189), (89, 190), (89, 191), (90, 189), (90, 190), (90, 191)],
          [(88, 192), (88, 193), (88, 194), (89, 192), (89, 193), (89, 194), (90, 192), (90, 193), (90, 194)],
          [(87, 197), (87, 198), (87, 199), (88, 197), (88, 198), (88, 199), (89, 197), (89, 198), (89, 199)],
          [(86, 189), (86, 190), (86, 191), (87, 189), (87, 190), (87, 191), (88, 189), (88, 190), (88, 191)],
          [(88, 195), (88, 196), (88, 197), (89, 195), (89, 196), (89, 197), (90, 195), (90, 196), (90, 197)],
          [(89, 192), (89, 193), (89, 194), (90, 192), (90, 193), (90, 194), (91, 192), (91, 193), (91, 194)],
          [(87, 195), (87, 196), (87, 197), (88, 195), (88, 196), (88, 197), (89, 195), (89, 196), (89, 197)],
          [(92, 183), (92, 184), (92, 185), (93, 183), (93, 184), (93, 185), (94, 183), (94, 184), (94, 185)],
          [(86, 191), (86, 192), (86, 193), (87, 191), (87, 192), (87, 193), (88, 191), (88, 192), (88, 193)],
          [(89, 189), (89, 190), (89, 191), (90, 189), (90, 190), (90, 191), (91, 189), (91, 190), (91, 191)],
          [(87, 195), (87, 196), (87, 197), (88, 195), (88, 196), (88, 197), (89, 195), (89, 196), (89, 197)]]

project_path = r'/Users/getang/Documents/EarthResearch/IIH/test/TemplateBuildingExample/BrainSlices'

num = -1
for sub in range(10, 21):
    print(sub)
    num += 1
    nii_path = os.path.join(project_path, f'OASIS-TRT-20-{sub}Slice121.nii.gz')
    new_value = random.sample(range(300, 401), 9)
    modified_img = modify_voxels_nii(nii_path, voxels[num], new_value)
    nib.save(modified_img, os.path.join(project_path, f'modified_OASIS-TRT-20-{sub}Slice121.nii.gz'))

# %%
nii_dir = r'/Users/getang/Documents/EarthResearch/IIH/test/TemplateBuildingExample/BrainSlices/'
nii_path = 
nii_img = nib.load(nii_path)
data = nii_img.get_fdata()