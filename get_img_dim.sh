#bin/bash

IMG_FILES="/home/junwon/smt-data/images_gray/SO8/*"



for f in $IMG_FILES
do
    echo $f
    identify $f
done


