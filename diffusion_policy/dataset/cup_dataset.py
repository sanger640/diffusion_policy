from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class CupDataset(BaseImageDataset):
    def __init__(self,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            shape_meta=None, # <--- ADDED THIS
            **kwargs         # <--- ADDED THIS to catch any other unused args
            ):
        
        super().__init__()
        
        # 1. Load Zarr into Memory
        # We explicitly request the keys we know exist in your dataset
        print(f"Loading ReplayBuffer from {dataset_path}...")
        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, 
            keys=['obs/agent_view_image', 'obs/agent_pos', 'action']
        )
        
        # 2. Train/Val Split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        
        # 3. Downsampling (Optional)
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed
        )

        # 4. Sequence Sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
        )
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        
        # Load raw data
        raw_action = self.replay_buffer['action'][:]
        raw_pos = self.replay_buffer['obs/agent_pos'][:]
        
        # --- FIX: Slice to 4D (Pos + Grip) ---
        # Keep indices 0,1,2 (XYZ) and 9 (Grip). Drop 3-8 (Rot).
        action_4d = np.concatenate([raw_action[:, :3], raw_action[:, 9:]], axis=-1)
        
        # Do the same for state if you want 4D inputs too (Recommended)
        pos_4d = np.concatenate([raw_pos[:, :3], raw_pos[:, 9:]], axis=-1)
        
        data = {
            'action': action_4d,
            'agent_pos': pos_4d
        }
        # -------------------------------------
        
        # Fit normalizer on the new 4D data
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # Create image normalizer
        normalizer['agent_view_image'] = get_image_range_normalizer()
        return normalizer
    # def get_normalizer(self, mode='limits', **kwargs):
    #     normalizer = LinearNormalizer()
        
    #     data = {
    #         'action': self.replay_buffer['action'],
    #         'agent_pos': self.replay_buffer['obs/agent_pos']
    #     }
        
    #     # Fit standard normalizer first
    #     normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
    #     # --- CUSTOM ROTATION HANDLING ---
    #     for key in ['action', 'agent_pos']:
    #         # 1. Get parameters from the fitted normalizer
    #         params = normalizer[key].params_dict
            
    #         # 2. Disable normalization for Rot6D (indices 3-8)
    #         params['scale'][3:9] = 1.0
    #         params['offset'][3:9] = 0.0
            
    #         # 3. Create manual normalizer
    #         # FIX: Use .get_input_stats() instead of .input_stats
    #         normalizer[key] = SingleFieldLinearNormalizer.create_manual(
    #             scale=params['scale'],
    #             offset=params['offset'],
    #             input_stats_dict=normalizer[key].get_input_stats() # <--- FIXED
    #         )
            
    #     normalizer['agent_view_image'] = get_image_range_normalizer()
    #     return normalizer

    def __len__(self) -> int:
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        
        # Image
        image = sample['obs/agent_view_image']
        # image = np.moveaxis(image, -1, 1).astype(np.float32) / 255.0
        image = image.astype(np.float32) / 255.0
        
        # Raw 10D Data
        raw_pos = sample['obs/agent_pos'].astype(np.float32)
        raw_action = sample['action'].astype(np.float32)

        # --- FIX: Slice to 4D ---
        # Input: (T, 10) -> Output: (T, 4)
        pos_4d = np.concatenate([raw_pos[:, :3], raw_pos[:, 9:]], axis=-1)
        action_4d = np.concatenate([raw_action[:, :3], raw_action[:, 9:]], axis=-1)
        # ------------------------

        data = {
            'obs': {
                'agent_view_image': image,
                'agent_pos': pos_4d, # Now 4D
            },
            'action': action_4d      # Now 4D
        }
        return dict_apply(data, torch.from_numpy)
    # def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    #     sample = self.sampler.sample_sequence(idx)
        
    #     # 1. Process Image
    #     # Format: (T, H, W, C) -> (T, C, H, W)
    #     image = sample['obs/agent_view_image']
    #     # image = np.moveaxis(image, -1, 1)
    #     # Convert uint8 (0-255) to float32 (0-1)
    #     image = image.astype(np.float32) / 255.0
        
    #     # 2. Process State & Action
    #     agent_pos = sample['obs/agent_pos'].astype(np.float32)
    #     action = sample['action'].astype(np.float32)

    #     data = {
    #         'obs': {
    #             'agent_view_image': image, # Shape: [T, 3, 240, 320]
    #             'agent_pos': agent_pos,    # Shape: [T, 10]
    #         },
    #         'action': action               # Shape: [T, 10]
    #     }
    #     return dict_apply(data, torch.from_numpy)
    
def test():
    import os
    zarr_path = os.path.expanduser('/home/sanger/diffusion_policy/data/paper_implementation.zarr')
    dataset = CupDataset(zarr_path, horizon=16)
    print(dataset[2])

if __name__ == "__main__":
    test()