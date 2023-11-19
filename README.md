# COMP9444 Project - Image Caption Generation Model Optimization

## 9444-pro Max - Group members

|      NAME      | Student ID |
|:--------------:|:----------:|
|  Siyuan Wu   |  z5412156 |
| Yinru Sun | z5496029 |
| Ruowen Ma | z5364549 |
| Huangsheng Shi | z5431671 |
| Haonan Peng |  z5391150 |


## Preparation for running the file locally
Due to the lengthy duration of the training and feature engineering processes, all intermediate 
results are documented in the Colab's notebook. However, if a complete local run is 
desired, the following execution order must be followed:


## 1. execute the instruction in model part in notebook to download CLIP_prefix_caption folder:
!git clone https://github.com/rmokady/CLIP_prefix_caption && cd CLIP_prefix_caption

so the directory tree will be:
```tree
content
|_ _ CLIP_prefix_caption
|	|_ _ Images
|	|_ _data
|	|_ _notebooks		
|	|	
|	|	
|	|	
|	|..........		
|	|parse_coco.py		
|	|	
|	|	
|	|..........	
|	|train.py	
|	|	
```

## 2.download necessary packages using instructions in colab notebook:
!pip install scikit-image==0.19.0 ftfy regex tqdm
!pip install --upgrade transformers tokenizers
!pip install git+https://github.com/openai/CLIP.git
!pip install sacrebleu
!pip install Rouge


## 3. download and upload coco dataset to the specific directory in content using these instructions:

#Create necessary directories
!mkdir -p data/coco/annotations

#Download train_captions to data/coco/annotations.

!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/coco/annotations/annotations_trainval2017.zip
!unzip data/coco/annotations/annotations_trainval2017.zip -d data/coco/annotations/

#Download and unzip training and validation images

!wget http://images.cocodataset.org/zips/train2017.zip -O data/coco/training_images.zip
!wget http://images.cocodataset.org/zips/val2017.zip -O data/coco/validation_images.zip
!unzip data/coco/training_images.zip -d data/coco/
!unzip data/coco/validation_images.zip -d data/coco/


## 4.replace the content of the original train.py and parse_coco.py in CLIP_prefix_caption folder with our own train.py and parse_coco.py in .zip:

.zip tree structure:

.ipynb_checkpoints

.gitignore

CLIP_Project.ipynb

README.md

modified_res50_clip train.py

parse_coco_1.py

parse_coco_CBAM_res50.py

train.py

#Here parse_coco_1.py is paired with train.py for running all image models(RN50、RN101、RN50x4、Vit).Similarly, modified_res50_clip is compatible with train.pyparse_coco_CBAM_res50.py for just RN50 with cbam， but other models with cbam are not included here.


## 5. run the parse_coco.py and train.py in sequence:

#this is parse training data, RN101 RN50x4 Vit-B/32  model selected as needed, but Vit-B/32 is used for DAT not cbam, and it is incompatible with cbam

!python CLIP_prefix_caption/parse_coco.py --clip_model_type RN50   

#this is parse validation data, RN101 RN50x4 Vit-B/32  model selected as needed, but Vit-B/32 is used for DAT not cbam, and it is incompatible with cbam

!python CLIP_prefix_caption/parse_coco.py --clip_model_type RN50 --is_val True #this is parse validation data


#train the model using different names' pkl generated by parse_coco.py, getting results into target directory

!python /content/CLIP_prefix_caption/train.py --data /content/data/coco/oscar_split_RN50_train.pkl --val_data /content/data/coco/oscar_split_RN50_val.pkl --out_dir /content/coco_train/


## 6. collect data and plot them in any way as you like.










