
"""
Usage:
python eval_dummy.py --checkpoint data/outputs/2025.12.03/latest.ckpt
"""

import sys
import os
import click
import time
import hydra
import torch
import dill
import numpy as np
from omegaconf import OmegaConf

@click.command()
@click.option('-c', '--checkpoint', required=True, help='Path to the .ckpt file')
@click.option('-d', '--device', default='cuda:0', help='Device to run inference on')
def main(checkpoint, device):
    # 1. Load Checkpoint
    # We use dill because Hydra configs often contain complex python objects
    print(f"Loading checkpoint: {checkpoint}")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # 2. Reconstruct the Workspace (The Training Container)
    # We essentially "re-hydrate" the class used for training
    cls = hydra.utils.get_class(cfg._target_)
    # We pass a dummy output dir because the workspace constructor requires it, 
    # even though we won't write to it.
    workspace = cls(cfg, output_dir="data/dummy_eval_output") 
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 3. Extract the Policy (The Neural Network)
    # If EMA was used, we load the smoothed weights (better performance)
    policy = workspace.model
    if cfg.training.use_ema:
        print("Using EMA model weights.")
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    policy.num_inference_steps = 8
    
    # 4. Generate Dummy Data (Based on Config Shapes)
    # We look at the config stored INSIDE the checkpoint to know what shapes to create.
    print("\n--- Constructing Dummy Observation ---")
    
    obs_shape_meta = cfg.task.shape_meta.obs
    n_obs_steps = cfg.n_obs_steps # e.g., 2
    batch_size = 1
    
    obs_dict = {}
    
    for key, attr in obs_shape_meta.items():
        # Shape from config is [C, H, W] or [D]
        # Policy expects [Batch, Time, ...] -> [1, 2, ...]
        raw_shape = tuple(attr.shape)
        print("shape")
        print(raw_shape)
        input_shape = (batch_size, n_obs_steps) + raw_shape
        
        print(f"Creating dummy input for '{key}': {input_shape}")
        
        if attr.get('type') == 'rgb':
            # Create random images (0.0 to 1.0)
            data = torch.rand(input_shape, dtype=torch.float32, device=device)
        else:
            # Create random low-dim state
            data = torch.randn(input_shape, dtype=torch.float32, device=device)
            
        obs_dict[key] = data

    # 5. Run Inference
    print("\n--- Running Inference ---")
    # while(True):
    with torch.no_grad():
        # This runs the full Denoising Loop (e.g., 8 steps)
        start_time = time.time()
        result = policy.predict_action(obs_dict)
        print(time.time() - start_time)
        
    # # 6. Print Results
    action = result['action'] # Shape: [Batch, Horizon, Action_Dim]
    action_np = action.detach().cpu().numpy()
    
    print("\n✅ Inference Successful!")
    print(f"Output Action Shape: {action_np.shape} (Batch, Horizon, Dim)")
    print("First 5 steps of the predicted trajectory:")
    # Print first batch, first 5 steps
    print(action_np[0, :5, :])

if __name__ == '__main__':
    main()