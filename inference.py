import sys
import os
import torch
import cv2
import numpy as np
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.common.pytorch_util import dict_apply

# --- Configuration ---
# Update this to match the specific checkpoint you want to load
CHECKPOINT_PATH = "data/outputs/YOUR_TIMESTAMP_FOLDER/checkpoints/latest.ckpt"
# ---------------------

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('diffusion_policy', 'config')),
    config_name="train_franka_cup" # Use the same config name you trained with
)
def main(cfg: OmegaConf):
    # 1. Resolve config
    OmegaConf.resolve(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Instantiate Policy
    print("Loading Policy...")
    policy = hydra.utils.instantiate(cfg.policy)
    
    # 3. Load Checkpoint
    # The checkpoint contains the full workspace state (model, optimizer, epoch)
    # We only need the model weights ('state_dict').
    payload = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    # The workspace wraps the policy in a 'model' attribute, so the keys start with 'model.'
    # We need to remove this prefix to load it into the policy directly.
    state_dict = payload['state_dict']
    policy_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    
    # Load weights
    policy.load_state_dict(policy_state_dict)
    policy.to(device)
    policy.eval()
    print("Policy Loaded Successfully!")

    # 4. Prepare Dummy Observation
    # In a real scenario, you would grab these from your RealSense camera and Robot SDK.
    # Here we create random data matching your shapes.
    
    # Config parameters
    n_obs_steps = cfg.n_obs_steps # 2
    img_h, img_w = 240, 320       # Or 96, 96 if you resized
    action_dim = 10
    
    # Create a batch of 1 (B=1)
    # Obs shape: (B, T, C, H, W)
    dummy_img = torch.zeros((1, n_obs_steps, 3, img_h, img_w), dtype=torch.float32).to(device)
    
    # State shape: (B, T, D)
    dummy_pos = torch.zeros((1, n_obs_steps, action_dim), dtype=torch.float32).to(device)

    obs_dict = {
        'agent_view_image': dummy_img,
        'agent_pos': dummy_pos
    }

    # 5. Run Inference
    print("Running Inference...")
    with torch.no_grad():
        # The policy takes dictionary input and returns dictionary output
        result = policy.predict_action(obs_dict)
        
    # 6. Extract Action
    # The result is (B, Horizon, Action_Dim) -> e.g., (1, 16, 10)
    action_sequence = result['action'][0].cpu().numpy()
    
    print("\n--- Inference Result ---")
    print(f"Input Image Shape: {dummy_img.shape}")
    print(f"Predicted Trajectory Shape: {action_sequence.shape}")
    print("First 3 predicted steps (Position [x,y,z]):")
    print(action_sequence[:3, :3]) # Print first 3 steps, XYZ only

if __name__ == "__main__":
    main()