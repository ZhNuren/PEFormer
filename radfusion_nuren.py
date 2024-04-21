import pandas as pd
from torch.utils.data import Dataset
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence

class RadFusionCT(Dataset):
    def __init__(self, pkl_ct_file, pkl_emr_file, csv_file, split, mode, transform=None):
        with open(pkl_ct_file, 'rb') as f:
            self.pkl = pickle.load(f)

        with open(pkl_emr_file, 'rb') as f:
            self.emr = pickle.load(f)

        self.max_emb = 110
        self.csv = pd.read_csv(csv_file)
        self.csv = self.csv.sort_values(by=['study_num', 'slice_idx'])
        self.transform = transform
        self.split = split
        self.mode = mode
        self.study_num = []
        self.data = []
        cur_id = -1

        for i in range(len(self.csv)):
            if self.csv['split'][i] != self.split:
                continue
            if i == 0 or self.csv['study_num'][i] != self.csv['study_num'][i-1]:
                self.study_num.append(self.csv['study_num'][i])
                cur_id += 1
                self.data.append({
                    "embeddings": [self.emr[self.csv['study_num'][i]]] + [self.pkl[int(f"{self.csv['study_num'][i]}")]],
                    "label": self.csv['target'][i]
                })
            
        print(f"Loaded {len(self.data)} samples for {split} set.")
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = int(self.data[idx]["label"])
        embeddings = self.data[idx]["embeddings"]
        
        # Convert list of embeddings to tensors
        embeddings = [torch.tensor(e) for e in embeddings]

        # Pad all embeddings to the maximum length in this batch to ensure they can be stacked
        # Specify padding value if necessary, e.g., -1 or 0
        embeddings = torch.stack(embeddings, dim =0)
        # padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)
        
        # Separate ehr_embedding (first tensor) and ct_embeddings (rest of tensors)
        ehr_embedding = embeddings[0]
        ct_embeddings = embeddings[1:].squeeze(0)
        

        # averaged_ct_embedding = torch.mean(ct_embeddings, dim=0)
        # max_pooled_ct_embedding, _ = torch.max(ct_embeddings, dim=0)
        
        # print(ehr_embedding.shape)
        # print(averaged_ct_embedding.shape)
        # print('\n')
        if self.mode == 'ehr':
            return ehr_embedding, target
        elif self.mode == 'ct':
            return ct_embeddings, target
        else:  
            return ehr_embedding, ct_embeddings, target
