from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader


class FusionDataset:
    def __init__(self, train_loader: DataLoader, skeleton_embeddings: Dict, device: str,  val_loader: DataLoader = None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.clips_dict = {}
        self.skeleton_embeddings = skeleton_embeddings
        self.device = device
    
    def generate_clips_dict(self):
        pbar = tqdm(self.train_loader, desc="", total=len(self.train_loader))
        for batch in pbar:
            # Gather batch and move to device
            clips, _, paths = batch['clips'].to(self.device), batch['labels'].to(self.device), batch['paths']
            paths = [path.split('/')[-1] for path in paths]
            self.clips_dict.update({paths[i]: clips[i] for i in range(len(paths))})

    def fusion_tuples_train(self, key: str):
        skeleton_emb = self.skeleton_embeddings[key]
        clip_tensor = self.clips_dict[key]
        return skeleton_emb, clip_tensor
