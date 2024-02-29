#!/bin/bash

# Loop through each subject and session to organize files
for folder in $(ls $current_dataset_dir); do
    # Extract subject and session from the folder name
    # Assuming folder names are in the format "sub-XX_ses-YY"
    subject=$(echo $folder | cut -d'_' -f1)
    session=$(echo $folder | cut -d'_' -f2)
    
    # Define the source directory
    src_dir="$current_dataset_dir/$folder"
    
    # Define the destination directory for anatomical images
    dest_anat_dir="$bids_dir/$subject/$session/anat"
    dest_derivative_dir="$bids_dir/derivatives/$subject/$session/anat"
    
    # Create the destination directory structure
    mkdir -p $dest_anat_dir
    mkdir -p $dest_derivative_dir/Transforms
    mkdir -p $dest_derivative_dir/Metrics
    
    # Find and move T1w MRI images to the appropriate BIDS directory, renaming them
    t1w_files=$(find $src_dir -type f -name "Denoised_*T1w.nii")
    for t1w_file in $t1w_files; do
        mv "$t1w_file" "$dest_anat_dir/${subject}_${session}_T1w.nii"
    done
    
    # Assuming a similar naming convention for T2w images, adjust as needed
    t2w_files=$(find $src_dir -type f -name "Denoised_*T2w.nii")
    for t2w_file in $t2w_files; do
        mv "$t2w_file" "$dest_anat_dir/${subject}_${session}_T2w.nii"
    done

    t2weye_files=$(find $src_dir -type f -name "Denoised_*T2wEye.nii")
    for t2weye_file in $t2weye_files; do
        mv "$t2weye_file" "$dest_anat_dir/${subject}_${session}_acq-Eye_T2w.nii"
    done

    derivative_files=$(find $src_dir -type f -name "trf*")
    for derivative_file in $derivative_files; do
        mv "$derivative_file" "$dest_derivative_dir/Transforms/$(basename $derivative_file)"
    done

    for derivative_file in $(find $src_dir/Metrics -type f); do
        echo $derivative_file
        mv "$derivative_file" "$dest_derivative_dir/Metrics/$(basename $derivative_file)"
    done

done

echo "Dataset organized into BIDS format."

# Running the docker container
docker run -it --rm -v /mnt/d/users/getang/tmp/mydata_bids:/data:ro -v /mnt/d/users/getang/tmp/report:/out nipreps/mriqc:latest /data /out participant --participant_label 15 --verbose True

docker run -it --rm -v /mnt/d/users/getang/tmp/mydata_bids:/data:ro -v /mnt/d/users/getang/tmp/bids_report:/out nipreps/mriqc:latest /data /out group
