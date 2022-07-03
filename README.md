# Heatmap branch

## 1. Environment
```
pip install -r requirement.txt
```

## 2. Train
* Revise the data folder ```DATA_DIR```in files under ```scripts_sh``` folder.
* Train using scripts in ```scripts_sh``` folder, e.g.
```
sh scripts_sh/tiny-imagenet/resnet50_dct/train_baseline.sh
```

## 3. Calculate heatmaps
```
sh scripts_sh/tiny-imagenet/resnet50_dct/calculate_heatmap.sh
```
* For training set, we set ```is_train``` in ```data/build.py``` ```build_transformer``` to ```False``` to disable data augmentation. 


* Checkpoint can be downloaded from [Google drive](https://drive.google.com/file/d/1igQAH-pTN8ieh0akgp2Khwt-SNj0mDWq/view?usp=sharing)

## 4. Visualization
```
python draw_sns.py
```
