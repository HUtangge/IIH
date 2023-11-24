#!/bin/bash
dim=3 # image dimensionality
AP="/mnt/d/users/getang/Installed_apps/opt/ANTs/bin/" # /home/yourself/code/ANTS/bin/bin/  # path to ANTs binaries
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2  # controls multi-threading
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS
f=$1 ; m=$2    # fixed and moving image file names
if [[ ! -s $f ]] ; then echo no fixed $f ; exit; fi
if [[ ! -s $m ]] ; then echo no moving $m ;exit; fi
nm1=` basename $f | cut -d '.' -f 1 `
nm2=` basename $m | cut -d '.' -f 1 `
echo $nm1
reg=${AP}antsRegistration           # path to antsRegistration
echo affine $m $f outname is $nm
its=10000x1000x100
pad=20

ImageMath 3 pad1.nii.gz  PadImage $m  $pad 
ThresholdImage 3 pad1.nii.gz thresh1.nii.gz Otsu 3 
ThresholdImage 3 thresh1.nii.gz thresh1.nii.gz 2 3 
ThresholdImage 3 $m          mask2.nii.gz Otsu 3 
ThresholdImage 3 mask2.nii.gz mask2.nii.gz 2 3 

ThresholdImage 3 eye.nii.gz mask2.nii.gz 0.1 Inf
m=eye.nii.gz
f=t.nii.gz
antsRegistration --verbose 1 --dimensionality 3 --float 0 \
  --output [2b,2bWarped.nii.gz,2bInverseWarped.nii.gz] \
  --interpolation Linear --use-histogram-matching 0 \
  --winsorize-image-intensities [0.005,0.995] \
   --initial-moving-transform [$f,$m,1] \
 --transform translation[0.1] --metric MI[$f,$m,1,32,Random,0.25] \
--convergence [50,1e-6,10] --shrink-factors 1 --smoothing-sigmas 0vox \
 --transform Rigid[0.1] --metric MI[$f,$m,1,32,Random,0.25] \
--convergence [500x250x50,1e-6,10] --shrink-factors 2x2x1 --smoothing-sigmas 2x1x0vox \
-x [mask2.nii.gz,mask2.nii.gz]