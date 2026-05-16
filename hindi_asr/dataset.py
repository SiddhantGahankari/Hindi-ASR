import json
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler


MANIFEST_PATH_TRAIN = "data/manifest_train.jsonl"
MANIFEST_PATH_VAL = "data/manifest_val.jsonl"
STATS_PATH = "data/stats/global_stats.npz"
BATCH_SIZE = 16
BUCKET_BOUNDARIES = [3, 6, 9, 12]

class HindiASRDataset(Dataset):
    def __init__(self, manifest_path, stats_path, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.entries = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                self.entries.append(json.loads(line.strip()))
        stats = np.load(stats_path)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(np.sqrt(stats["var"] + 1e-9), dtype=torch.float32)
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, index):
        entry = self.entries[index]
        mel = torch.tensor(np.load(entry["mel_path"]).astype(np.float32))
        mel = (mel - self.mean[:, None]) / self.std[:, None]
        if self.is_train:
            mel = mel.unsqueeze(0)
            t_max = min(100, mel.shape[-1] // 2)
            freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=27)
            time_mask = torchaudio.transforms.TimeMasking(time_mask_param=t_max, iid_masks=False)
            mel = freq_mask(mel)
            mel = freq_mask(mel)
            mel = time_mask(mel)
            mel = time_mask(mel)
            mel = mel.squeeze(0)
        tokens = torch.tensor(entry["tokens"], dtype=torch.long)
        duration = entry["duration"]
        return mel, tokens, duration

class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size, boundaries, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.shuffle = shuffle
        
        # Organize indices into buckets based on duration
        self.buckets = self._build_buckets()
        
        # Pre-calculate the total number of batches
        self.num_batches = self._calculate_num_batches()

    def _build_buckets(self):
        # Create (len(boundaries) + 1) buckets
        buckets = [[] for _ in range(len(self.boundaries) + 1)]
        
        for idx, entry in enumerate(self.dataset.entries):
            duration = entry["duration"]
            assigned = False
            for b, boundary in enumerate(self.boundaries):
                if duration <= boundary:
                    buckets[b].append(idx)
                    assigned = True
                    break
            if not assigned:
                buckets[-1].append(idx)
        return buckets

    def _calculate_num_batches(self):
        count = 0
        for bucket in self.buckets:
            # Add number of batches this bucket will produce
            count += (len(bucket) + self.batch_size - 1) // self.batch_size
        return count

    def __iter__(self):
        all_batches = []
        
        for bucket in self.buckets:
            if len(bucket) == 0:
                continue
            
            indices = bucket.copy()
            if self.shuffle:
                random.shuffle(indices)
            
            # Create batches from the current bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) > 0:
                    all_batches.append(batch)
        
        # Shuffle the order of batches to ensure diverse training
        if self.shuffle:
            random.shuffle(all_batches)
            
        for batch in all_batches:
            yield batch

    def __len__(self):
        return self.num_batches
    
def collate_fn(batch):
    mels, tokens, durations = zip(*batch)
    mel_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    token_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    max_mel_len = max(m.shape[1] for m in mels)
    max_token_len = max(len(t) for t in tokens)
    padded_mels = torch.zeros(len(mels), 80, max_mel_len)
    padded_tokens = torch.zeros(len(tokens), max_token_len, dtype=torch.long)
    for i, (mel, token) in enumerate(zip(mels, tokens)):
        padded_mels[i, :, :mel.shape[1]] = mel
        padded_tokens[i, :len(token)] = token
    return padded_mels, padded_tokens, mel_lengths, token_lengths
