
import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# Default transforms: resize to 224x224 and normalize
train_transform = T.Compose([
    T.Resize((224, 224)),   
    T.ToTensor(),  # convert image to tensor 
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]), # normalize the image 
])

eval_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class HatefulMemesDataset(Dataset):
    

    def __init__(self, data_root, split="train", transform=None):
        # initialize the dataset with the data root, split, and transform
        # and set the data root, split, and transform variables   
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = transform

        # set the jsonl path based on the split
        if split == "train":
            jsonl_path = os.path.join(data_root, "train.jsonl")
        elif split == "dev":
            jsonl_path = os.path.join(data_root, "dev.jsonl")
        elif split == "test":
            jsonl_path = os.path.join(data_root, "test.jsonl")
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Could not find {jsonl_path}")

        self.samples = [] # initialize the samples list
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip() # remove the whitespace from the line
                if not line:
                    continue
                obj = json.loads(line) # load the json object from the line
                img_rel = obj["img"]
                img_path = os.path.join(data_root, img_rel)  
                label = obj.get("label", None) # get the label from the object
                self.samples.append({
                    "id": obj["id"], # add the id to the sample
                    "img_path": img_path, # add the image path to the sample
                    "label": label, # add the label to the sample
                    "text": obj.get("text", ""), # add the text to the sample
                })

    def __len__(self):
        return len(self.samples) # return the length of the samples list

    def __getitem__(self, idx):
        sample = self.samples[idx] # get the sample at the given index
        img = Image.open(sample["img_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img) # apply the transform to the image

        label = sample["label"] # get the label from the sample
        # For test set where label may not be available
        if label is None:
            return img, -1, sample["id"] # return the image, -1, and the id

        return img, int(label), sample["id"]
