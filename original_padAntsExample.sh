#!/bin/bash
dim=3 # image dimensionality
AP="" # /home/yourself/code/ANTS/bin/bin/  # path to ANTs binaries
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2  # controls multi-threading
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS
f=$1 ; m=$2    # fixed and moving image file names
if [[ ! -s $f ]] ; then echo no fixed $f ; exit; fi
if [[ ! -s $m ]] ; then echo no moving $m ;exit; fi
nm1=` basename $f | cut -d '.' -f 1 `
nm2=` basename $m | cut -d '.' -f 1 `
reg=${AP}antsRegistration           # path to antsRegistration
echo affine $m $f outname is $nm
its=10000x1000x100
pad=10
ImageMath 3 pad1.nii.gz  PadImage $f  $pad 
ThresholdImage 3 pad1.nii.gz thresh1.nii.gz Otsu 3 
ThresholdImage 3 thresh1.nii.gz thresh1.nii.gz 2 3 
ThresholdImage 3 $m          thresh2.nii.gz Otsu 3 
ThresholdImage 3 thresh2.nii.gz thresh2.nii.gz 2 3 
$reg -d $dim -r [ thresh1.nii.gz , thresh2.nii.gz ,1]  \
                        -m mattes[ pad1.nii.gz , $m , 1 , 32, regular, 0.2 ] \
                         -t affine[ 0.1 ] \
                         -c [$its,1.e-8,10]  \
                        -s 4x2x1vox  \
                        -f 6x4x2 -l 1 \
                        -m demons[  pad1.nii.gz, $m , 1 , 0 ] \
                         -t syn[ .2, 3, 0.0 ] \
                         -c [200x200x20x0,1.e-8,10]  \
                        -s 4x2x1x0vox  \
                        -f 8x4x2x1 -l 1 -u 1 -z 1 \
                       -o [${nm},${nm}_diff.nii.gz,${nm}_inv.nii.gz]

antsApplyTransforms -d 3 -i $m -o temp.nii.gz -r pad1.nii.gz -t ${nm}1Warp.nii.gz -t ${nm}0GenericAffine.mat
