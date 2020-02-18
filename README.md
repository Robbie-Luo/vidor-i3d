# I3D models trained on Vidor

## Overview

This repo is the trunk net 4 2nd task of VidVRD: Video Relation Prediction

The 2st stage project: Video Action Detection

[The Grand Challenge MM2019](http://lms.comp.nus.edu.sg/research/dataset.html) 

## Download

[Vidor](http://lms.comp.nus.edu.sg/research/dataset.html)

### Extract the Frames 
To load the VidOR dataset, we first need to extract the frames for each split by simply running the following script after you modify the dataset path in [frames.py](frames.py)
```
python frames.py -split=training
```

## Fine-tuning I3D
[train_i3d.py](train_i3d.py) 
contains the code to fine-tune I3D based on the details in the paper and obtained from the authors.

E.g.
```bash
python train_i3d.py 
```

[vidor_dataset.py](vidor_dataset.py) script <b>VidorDataset</b>
contains the code to load video segments for training.

[evaluate_i3d.py](evaluate_i3d.py) contains some codes for computing the model accuracy 

## Feature Extraction
[extract_features.py](extract_features.py) 
contains the code to load a pre-trained I3D model and extract the features 
and save the features as numpy arrays.

E.g.
```bash
python extract_features.py
```
The [vidor_dataset_full.py](vidor_dataset_full.py) script 
loads an entire video to 
extract per-segment features.



