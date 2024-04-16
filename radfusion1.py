import pandas as pd
from torch.utils.data import Dataset
import pickle
import torch
import random

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
        first_embedding = embeddings[0] if embeddings.size(0) > 0 else None

        if self.transform and random.random() < self.aug_prob:
            if embeddings.size(0) > 1:
                remaining_embeddings = embeddings[1:]

                if self.transform == 'flip':
                    remaining_embeddings = torch.flip(remaining_embeddings, [0])
                elif self.transform == 'shuffle':
                    indices = torch.randperm(remaining_embeddings.size(0))
                    remaining_embeddings = remaining_embeddings[indices]
                elif self.transform == 'flip_and_shuffle':
                    remaining_embeddings = torch.flip(remaining_embeddings, [0])
                    indices = torch.randperm(remaining_embeddings.size(0))
                    remaining_embeddings = remaining_embeddings[indices]

                embeddings = torch.cat([first_embedding.unsqueeze(0), remaining_embeddings], dim=0)
            else:
                # No remaining embeddings to flip or shuffle, just use the first_embedding
                embeddings = first_embedding.unsqueeze(0) if first_embedding is not None else torch.tensor([])

        src_seq_len = embeddings.size(0)
        if embeddings.size(0) < self.max_emb:
            pad = torch.zeros(self.max_emb - embeddings.size(0), embeddings.size(1))
            embeddings = torch.cat((embeddings, pad), dim=0)
        
        mask = torch.arange(self.max_emb) >= src_seq_len

        study_num = self.study_num[idx]

        return study_num, embeddings, mask, target


