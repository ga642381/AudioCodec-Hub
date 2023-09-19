import librosa
from torch.utils.data import Dataset
from typing import List
from transformers import EncodecFeatureExtractor

class SpeechDataset(Dataset):
    def __init__(self, file_paths:List, target_sr):
        self.file_paths = file_paths
        self.sr = target_sr
        self.processor = EncodecFeatureExtractor(sampling_rate=self.sr)

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        wav, _ = librosa.load(path, sr=self.sr)
        return wav, path
    
    def collate_fn(self, data):
        audios = [d[0] for d in data]
        paths = [d[1] for d in data]
        inputs = self.processor(raw_audio=audios, sampling_rate=self.sr, return_tensors="pt")
        return inputs['input_values'], inputs['padding_mask'], paths
