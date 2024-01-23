#!/bin/bash

# Archive the dataset
# Running at the Windows Command Prompt
"C:\Program Files\7-Zip\7z.exe" a -t7z "D:\users\Rawdata.7z" "D:\users\Rawdata\*"

# Define the base directory of your current dataset and the target BIDS directory
current_dataset_dir="/path/to/your/current/dataset"
bids_dir="/path/to/your/bids/dataset"

# Loop through each subject and session to organize files
for subject in $(ls $current_dataset_dir); do
    for session in $(ls $current_dataset_dir/$subject); do
        # Define the source and target directories
        src_dir="$current_dataset_dir/$subject/$session"
        dest_dir="$bids_dir/sub-$subject/ses-$session/anat"
        
        # Create the target directory structure
        mkdir -p $dest_dir
        
        # Assuming T1w and T2w MRI images are named clearly, move them to the BIDS structure
        mv "$src_dir/T1w.nii.gz" "$dest_dir/sub-${subject}_ses-${session}_T1w.nii.gz"
        mv "$src_dir/T2w.nii.gz" "$dest_dir/sub-${subject}_ses-${session}_T2w.nii.gz"
    done
done

echo "Dataset organized into BIDS format."
