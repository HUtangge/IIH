#!/bin/bash
bids_dir="/Users/getang/Documents/EarthResearch/IIH/IIH_MRIQC/out"
bids_dir_workstation="D:/users/getang/IIH/IIH_MRIQC_15_1/out"
local_key_file="/Users/getang/.ssh/id_rsa"
remote_host="getang@138.245.135.142"
# Loop through each subfolder in the parent directory
for subjectfolder in "$bids_dir"/*; do
    # Check if the current item is a directory
    if [ -d "$subjectfolder" ]; then
        # Find all .nii.gz files in the current subfolder
        for sesfolder in "$subjectfolder"/*; do
            for file in "$sesfolder"/anat/*T2w.json; do
                if [ -e "$file" ]; then
                    directory=$(dirname "$file")
                    filename=$(basename "$file")
                    new_filename="${filename/sub/Masked_retroorbital_sub}"
                    new_file="${directory}/${new_filename}"
                    local_file="${new_file%.json}.nii.gz"
                    workstation_file="${local_file/#$bids_dir/$bids_dir_workstation}" 
                    # echo "Processing $local_file"                   
                    if ssh "$remote_host" "if exist "$workstation_file" (exit 0) else (exit 1)"; then
                        echo "File $workstation_file copied successfully."
                        scp -i "$local_key_file" "$remote_host:$workstation_file" "$local_file"
                        fast -t 2 -v -n 2 -o "${directory}/segment_retroorbital_T2w" -g -S 1 "$local_file"
                    else
                        echo "File $workstation_file does not exist on the remote Windows station. Skipping."
                    fi
                fi
            done
        done
    fi
done

