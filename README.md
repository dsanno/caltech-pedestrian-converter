# Caltech Pedestrian Dataset Converter

This script converts `.seq` files into `.jpg` files, `.vbb` files into `.pkl` files from [Caltech Pedestrian Dataset](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/).

# Requirements
- Python 2.7
- NumPy 1.10.4
- SciPy 0.17.0

# How to use

## Download dataset

```
$ bash download.sh
```

## Convert dataset

```
$ python converte.py
```

# Output files

## Annotation

data/annotations.json

## Video frame images

data/set(set index)/V(video index)/(frame index).jpg

ex. data/images/set00/V000/00000.jpg
