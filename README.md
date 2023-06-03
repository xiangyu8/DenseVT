### This is an official repo for paper "Explicitly Increasing Input Information Density for Vision Transformers on Small Datasets".
 - version 1: Increasing input information density for vision transformers on small datasets, accepted as extended abstract by CVPR workshop (WiCV) 2022.
 - version 2: Explicitly Increasing Input Information Density for Vision Transformers on Small Datasets, accepted by NeurIPS workshop (VTTA) 2022.

* main branch: for classification with vision transformers.
* heatmap branch: to select channels based on heatmaps.

## 1. Environment
```
pip install -r requirement.txt
```

## 2. Train
* Revise the data folder ```DATA_DIR```in files under ```scripts_sh``` folder.
* Train using scripts in ```scripts_sh``` folder, e.g.
```
sh scripts_sh/swin_dct/dct/train_baseline_dct_flowers.sh
```

## 3. Test
* Checkpoints can be downloaded from [Google drive](https://drive.google.com/file/d/1Vzg3HQoQIrRAfFHm5DAPCk6LgZMHihMW/view?usp=sharing)
* Put checkpoints to corresponding folders and testing scripts are same with training.

## 4. Acknowledgments
Our codes are highly based on [VT-drloc](https://github.com/yhlleo/VTs-Drloc.git). 
