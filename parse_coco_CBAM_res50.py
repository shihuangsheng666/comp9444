import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import time  # Import the time module to measure elapsed time
import pandas as pd
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.num_channels = num_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels)
        )
        
    def forward(self, x):
        avg_pool = self.avg_pool(x).view(-1, self.num_channels)
        max_pool = self.max_pool(x).view(-1, self.num_channels)
        avg_out = self.shared_MLP(avg_pool)
        max_out = self.shared_MLP(max_pool)
        out = avg_out + max_out
        scale = torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdims=True)
        max_out, _ = torch.max(x, dim=1, keepdims=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ModifiedCLIP(torch.nn.Module):
    def __init__(self, original_clip_model, cbam):
        super(ModifiedCLIP, self).__init__()
        self.visual = original_clip_model.visual
        self.cbam = cbam
        #self.proj = original_clip_model.visual.proj

    def forward(self, image):
        x = self.visual.conv1(image)
        x = self.visual.bn1(x)
        x = self.visual.relu1(x)
        
        x = self.visual.conv2(x)
        x = self.visual.bn2(x)
        x = self.visual.relu2(x)
        
        x = self.visual.conv3(x)
        x = self.visual.bn3(x)
        x = self.visual.relu3(x)
        
        x = self.visual.avgpool(x)  # Using the avgpool layer as per your model structure
        
        # Continue with the subsequent layers
        x = self.visual.layer1(x)
        x = self.visual.layer2(x)
        x = self.visual.layer3(x)
        x = self.visual.layer4(x)

        # Apply CBAM after layer4
        x = self.cbam(x)

        # Assuming attnpool is a method of aggregating the features in the transformer architecture
        x = self.visual.attnpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        #x = self.visual.proj(x) 

        return x
    
    def encode_image(self, image):
        image = image.half()
        return self.forward(image)






def load_data(ann_list, part=5,clean = True):

    annotations = pd.DataFrame(ann_list)
    annotations = annotations.sort_values(by ='image_id')  
    num_to_sample = len(annotations['image_id'].unique())//part 
    sampled_image_ids = annotations['image_id'].drop_duplicates().sample(n=num_to_sample, random_state=11)

    sampled_annotations_df = annotations[annotations['image_id'].isin(sampled_image_ids)]
    clean_captions = sampled_annotations_df.to_dict(orient='records')
    if clean:
        table = str.maketrans('','',string.punctuation)  #remove all punctuation 
        for j in range(len(clean_captions)):        
            sentence = clean_captions[j]['caption'].replace("-"," ")
            # remove punctuation
            sentence = sentence.translate(table)
            split_sentence = sentence.split()
            # convert all to lower case
            split_sentence = [wrd.lower() for wrd in split_sentence]
            # remove 's and a that does not help with training
            split_sentence = [wrd for wrd in split_sentence if(len(wrd)>1)]
            # remove all string like numbers
            split_sentence = [wrd for wrd in split_sentence if(wrd.isalpha())]
            sentence = ' '.join(split_sentence)
            clean_captions[j]['caption'] = sentence
        return clean_captions  
    return clean_captions
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, CenterCrop, ToTensor, Normalize
import cv2

class CLAHETransform(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img = np.array(img)  
        if len(img.shape) == 2:  
            img = self.clahe.apply(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img)
            l = self.clahe.apply(l)
            img = cv2.merge((l, a, b))
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        img = Image.fromarray(img)  
        return img

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        RandomHorizontalFlip(0.2),
        RandomVerticalFlip(0.1),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
def main(clip_model_type: str, is_val: bool):
    start_time = time.time()  # Record the start time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model_name = clip_model_type.replace('/', '_')
    if not is_val: 
      out_path = f"./data/coco/oscar_split_{clip_model_name}_train.pkl"
    else:
      out_path = f"./data/coco/oscar_split_{clip_model_name}_val.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    preprocess = _transform(224)
    # Unfreeze layers starting from the last 'layer3' block and all 'layer4' blocks
    cbam = CBAM(2048) 

    clip_model = ModifiedCLIP(clip_model, cbam).to(device)
    for name, parameter in clip_model.visual.named_parameters():
       if 'layer3.5' in name or 'layer4' in name or 'attnpool' in name:
         parameter.requires_grad = True
       else:
         parameter.requires_grad = False  
    print("Opening the file...")
    if not is_val: 
      with open('./data/coco/annotations/annotations/captions_train2017.json', 'r', encoding='utf-8') as f:
          data = json.load(f)
    else:
      with open('./data/coco/annotations/annotations/captions_val2017.json', 'r', encoding='utf-8') as f:
          data = json.load(f)
    annotations = data.get('annotations', [])
    print(f"{len(annotations)} annotations in original file")
    annotations = load_data(annotations,5,True)
    print(f"{len(annotations)} annotations loaded from down sized json")
    all_embeddings = []
    all_captions = []
    for i, d in enumerate(tqdm(annotations)):
        img_id = d["image_id"]
        if not is_val:
            filename = f"./data/coco/train2017/{int(img_id):012d}.jpg"
        else:
            filename = f"./data/coco/val2017/{int(img_id):012d}.jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
          with torch.cuda.amp.autocast():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
            print(f"Saved {i + 1} embeddings and captions to {out_path}")

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    elapsed_time = time.time() - start_time  # Calculate the elapsed time
    print('Done')
    print(f"{len(all_embeddings)} embeddings saved")
    print(f"Total time: {elapsed_time:.2f} seconds")

    return 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--is_val', default=False)
    args = parser.parse_args()
    exit(main(args.clip_model_type,args.is_val))
