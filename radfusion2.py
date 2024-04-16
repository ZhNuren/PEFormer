import pandas as pd
from torch.utils.data import Dataset
import pickle
import torch
import random

class TransformCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, embeddings):
        first_embedding = embeddings[0]
        remaining_embeddings = embeddings[1:]
        for transform in self.transforms:
            remaining_embeddings = transform(remaining_embeddings)
        return torch.cat([first_embedding.unsqueeze(0), remaining_embeddings], dim=0)

class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, embeddings):
        if random.random() < self.p:
            return torch.flip(embeddings, [0])
        return embeddings

class RandomShuffle:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, embeddings):
        if random.random() < self.p:
            indices = torch.randperm(embeddings.size(0))
            return embeddings[indices]
        return embeddings

# Example usage:
# transforms = TransformCompose([
#     RandomFlip(p=0.5),
#     RandomShuffle(p=0.5)
# ])
class RadFusionCT(Dataset):
    def __init__(self, pkl_ct_file, pkl_emr_file, csv_file, split, transform=None, aug_prob=0.5):

        with open(pkl_ct_file, 'rb') as f:
            self.pkl = pickle.load(f)

        with open(pkl_emr_file, 'rb') as f:
            self.emr = pickle.load(f)
        self.max_emb = 110
        self.csv = pd.read_csv(csv_file)
        #sort by study_num
        self.csv = self.csv.sort_values(by=['study_num', 'slice_idx'])
        self.transform = transform
        self.split = split
        self.aug_prob = aug_prob
        self.study_num = []
        self.data = []
        cur_id = -1
        for i in range(len(self.csv)):
            if self.csv['split'][i] != self.split:
                continue
            if i == 0 or self.csv['study_num'][i] != self.csv['study_num'][i-1]:
                self.study_num.append(self.csv['study_num'][i])
                cur_id += 1
                self.data.append({"embeddings": [self.pkl[f"{self.csv['study_num'][i]}_{self.csv['slice_idx'][i]}"]], "label": self.csv['target'][i]})
                self.data[cur_id]["embeddings"].insert(0, self.emr[self.csv['study_num'][i]])

            else:
                self.data[cur_id]["embeddings"].append(self.pkl[f"{self.csv['study_num'][i]}_{self.csv['slice_idx'][i]}"])
        
        
        print(f"Loaded {len(self.data)} samples for {split} set.")
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = int(self.data[idx]["label"])
        embeddings = self.data[idx]["embeddings"]
        embeddings = torch.stack(embeddings, dim=0)

        # Check if there is more than one embedding to apply transformations on the remaining ones
        if embeddings.size(0) > 1:
            if self.transform:
                embeddings = self.transform(embeddings)

        # Padding embeddings to match the maximum embeddings size
        src_seq_len = embeddings.size(0)
        if embeddings.size(0) < self.max_emb:
            pad = torch.zeros(self.max_emb - embeddings.size(0), embeddings.size(1))
            embeddings = torch.cat((embeddings, pad), dim=0)
        
        mask = torch.arange(self.max_emb) >= src_seq_len

        study_num = self.study_num[idx]

        return study_num, embeddings, mask, target
