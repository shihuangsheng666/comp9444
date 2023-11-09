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
        ColorJitter(0.5,0.5,0.5),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
def main(clip_model_type: str, is_val: bool):
    start_time = time.time()  # Record the start time
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:  
        device = torch.device('cpu')
    clip_model_name = clip_model_type.replace('/', '_')
    if not is_val: 
      out_path = f"./data/coco/oscar_split_{clip_model_name}_train.pkl"
    else:
      out_path = f"./data/coco/oscar_split_{clip_model_name}_val.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    preprocess = _transform(224)
    print("Opening the file...")
    if not is_val: 
      with open('./data/coco/annotations/annotations/captions_train2017.json', 'r', encoding='utf-8') as f:
          data = json.load(f)
    else:
      with open('./data/coco/annotations/annotations/captions_val2017.json', 'r', encoding='utf-8') as f:
          data = json.load(f)
    annotations = data.get('annotations', [])
    print(f"{len(annotations)} annotations in original file")
    annotations = load_data(annotations,10,True)
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
