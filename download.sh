#! /bin/bash

if [ ! -d data ]; then
    mkdir data
fi
cd data

#wget http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/annotations.zip

#unzip annotations.zip
#rm -rf annotations.zip

for i in "00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10"; do
  wget http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set${i}.tar
  tar xvf set${i}.tar
  rm -rf set${i}.tar
done
