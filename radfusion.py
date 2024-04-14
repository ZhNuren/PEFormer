import pandas as pd
from torch.utils.data import Dataset
import pickle
import torch

class RadFusionCT(Dataset):
    def __init__(self, pkl_ct_file, pkl_emr_file, csv_file, split, transform=None):

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
        src_seq_len = embeddings.size(0)
        if embeddings.size(0) < self.max_emb:
            pad = torch.zeros(self.max_emb - embeddings.size(0), embeddings.size(1))
            embeddings = torch.cat((embeddings, pad), dim=0)
        
        mask = torch.arange(self.max_emb) >= src_seq_len

        study_num = self.study_num[idx]

        if self.transform:
            embeddings = self.transform(embeddings)
        
        return study_num, embeddings, mask, target

