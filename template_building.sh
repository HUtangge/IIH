
inputPath=${PWD}/
outputPath=${PWD}/Templateparallel_NormAvg/

export ANTSPATH=/mnt/c/users/tangge/ANTS/install/bin/

${ANTSPATH}/buildtemplateparallel.sh \
 -d 3 \
 -m 50x90x30 \
 -t GR \
 -s CC \
 -c 2 \
 -n 1 \
 -i 5 \
 -r 1 \
 -j 4 \
 -o ${outputPath}T_ \
 sub*.nii.gz