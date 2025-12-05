from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer, array_to_stats

class CupDataset(BaseImageDataset):
    def __init__(self,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            shape_meta=None,
            **kwargs
            ):
        
        super().__init__()
        
        print(f"Loading ReplayBuffer from {dataset_path}...")
        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, 
            keys=['obs/camera_1', 'obs/camera_2','obs/agent_pos', 'action']
        )
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed
        )

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
        
        # Load raw low-dim data to compute stats
        # We slice [:] to load into RAM, but we slice columns to keep 4D
        raw_action = self.replay_buffer['action'][:]
        raw_pos = self.replay_buffer['obs/agent_pos'][:]
        
        # Slice to 4D (Pos + Grip)
        action_4d = np.concatenate([raw_action[:, :3], raw_action[:, 9:]], axis=-1)
        pos_4d = np.concatenate([raw_pos[:, :3], raw_pos[:, 9:]], axis=-1)
        
        data = {
            'action': action_4d,
            'agent_pos': pos_4d
        }
        
        # Fit standard normalizer
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # Create manual normalizers (Identity for rotation is irrelevant now since we sliced it out,
        # but we reconstruct the object properly using array_to_stats)
        # for key in ['action', 'agent_pos']:
        #     stat = array_to_stats(data[key])
        #     params = normalizer[key].params_dict
            
        #     normalizer[key] = SingleFieldLinearNormalizer.create_manual(
        #         scale=params['scale'],
        #         offset=params['offset'],
        #         input_stats_dict=stat
        #     )
            
        # Image Normalizers for BOTH cameras
        normalizer['camera_1'] = get_image_range_normalizer()
        normalizer['camera_2'] = get_image_range_normalizer()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        
        # 1. Load Camera 1
        cam1 = sample['obs/camera_1'] # (T, 3, H, W) stored on disk
        cam1 = cam1.astype(np.float32) / 255.0
        
        # 2. Load Camera 2
        cam2 = sample['obs/camera_2']
        cam2 = cam2.astype(np.float32) / 255.0
        
        # 3. Load & Slice Low-Dim Data
        raw_pos = sample['obs/agent_pos'].astype(np.float32)
        raw_action = sample['action'].astype(np.float32)

        # Slice to 4D (Pos + Grip)
        pos_4d = np.concatenate([raw_pos[:, :3], raw_pos[:, 9:]], axis=-1)
        action_4d = np.concatenate([raw_action[:, :3], raw_action[:, 9:]], axis=-1)

        data = {
            'obs': {
                'camera_1': cam1,
                'camera_2': cam2,
                'agent_pos': pos_4d,
            },
            'action': action_4d
        }
        return dict_apply(data, torch.from_numpy)