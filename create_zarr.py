import os
import json
import numpy as np
import cv2
import zarr
from tqdm import tqdm
from glob import glob
from scipy.spatial.transform import Rotation as R
from numcodecs import Blosc # <--- Fixes the "no attribute Blosc" error

# --- Configuration ---
DATA_ROOT = "./episodes/episodes"  # Where your episode_0, episode_1 folders are
OUTPUT_ZARR = "data/paper_implementation.zarr"
RESIZE_W, RESIZE_H = 320, 240 # Resolution for Real-World Tasks

def quat_to_rot6d(quat_list):
    """
    Converts list of quaternions [x,y,z,w] to [r1...r6]
    """
    r = R.from_quat(quat_list)
    matrices = r.as_matrix() # (N, 3, 3)
    # Take first two columns and flatten
    rot6d = matrices[:, :, :2].reshape(-1, 6)
    return rot6d

def create_dataset():
    if os.path.exists(OUTPUT_ZARR):
        print(f"Dataset {OUTPUT_ZARR} already exists. Please delete it first.")
        return

    # 1. Collect Episodes
    episode_dirs = sorted(glob(os.path.join(DATA_ROOT, "*")))
    
    all_imgs = []
    all_states = []
    all_actions = []
    episode_ends = []

    print(f"Processing {len(episode_dirs)} episodes into Robomimic format...")

    for ep_dir in tqdm(episode_dirs):
        # Load Trajectory JSON
        json_files = glob(os.path.join(ep_dir, "*.json"))
        if not json_files: continue
        
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        waypoints = data['waypoints']
        
        # Load Images
        rgb_dir = os.path.join(ep_dir, "rgb_frames")
        img_files = sorted(glob(os.path.join(rgb_dir, "*.png")))
        
        # Parse timestamps from filenames (assuming milliseconds)
        img_timestamps = []
        for f in img_files:
            fname = os.path.basename(f).replace('.png', '')
            try:
                ts = float(fname) / 1000.0
            except ValueError:
                continue # Skip bad files
            img_timestamps.append(ts)
        img_timestamps = np.array(img_timestamps)

        # --- Process Trajectory ---
        positions = []
        quats = []
        grippers = []
        timestamps = []

        for wp in waypoints:
            positions.append(wp['position'])
            quats.append(wp['orientation'])
            grippers.append([1.0] if wp['gripper'] else [-1.0])
            timestamps.append(wp['timestamp'])

        # Convert Rotation
        rot6d = quat_to_rot6d(quats)
        
        # Construct State Vector: [Pos(3) + Rot6D(6) + Grip(1)] = 10 Dim
        states_np = np.concatenate([
            np.array(positions), 
            rot6d, 
            np.array(grippers)
        ], axis=1).astype(np.float32)

        # --- Process Images (Sync) ---
        ep_imgs = []
        for t_robot in timestamps:
            # Find closest image by timestamp
            diffs = np.abs(img_timestamps - t_robot)
            idx = np.argmin(diffs)
            
            img_path = img_files[idx]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (RESIZE_W, RESIZE_H))
            # HWC -> CHW (Channel First)
            img = np.transpose(img, (2, 0, 1)) 
            ep_imgs.append(img)

        # For Diffusion Policy, Action = Next State
        # We store the state sequence; the dataloader handles the horizon shifting
        all_imgs.append(np.array(ep_imgs))
        all_states.append(states_np)
        all_actions.append(states_np)
        episode_ends.append(len(states_np))

    # --- Write to Zarr ---
    # Concatenate all lists into massive arrays
    combined_imgs = np.concatenate(all_imgs, axis=0)
    combined_states = np.concatenate(all_states, axis=0)
    combined_actions = np.concatenate(all_actions, axis=0)
    episode_ends = np.cumsum(episode_ends)

    print("Writing to disk... (This might take a moment)")
    root = zarr.open(OUTPUT_ZARR, mode='w')
    
    # 1. Create Structure
    data_group = root.create_group('data')
    obs_group = data_group.create_group('obs')
    meta_group = root.create_group('meta')
    
    # 2. Add "Blessing" Metadata (Crucial for Robomimic Loader)
    # Must cast numpy int to python int for JSON serialization
    total_steps = int(episode_ends[-1])
    root.attrs["total"] = total_steps
    root.attrs["env_args"] = json.dumps({"env_name": "FrankaReal", "env_version": 1.0})
    root.attrs["layout"] = "robomimic"

    # 3. Create Datasets with Compression
    compressor = Blosc(cname='zstd', clevel=3, shuffle=1)
    
    # Images -> data/obs/agent_view_image
    obs_group.create_dataset('agent_view_image', data=combined_imgs, 
                             chunks=(100, 3, RESIZE_H, RESIZE_W), compressor=compressor)
    
    # State -> data/obs/agent_pos
    obs_group.create_dataset('agent_pos', data=combined_states, 
                             chunks=(100, 10), compressor=compressor)
    
    # Action -> data/action
    data_group.create_dataset('action', data=combined_actions, 
                              chunks=(100, 10), compressor=compressor)
    
    # Meta -> meta/episode_ends
    meta_group.create_dataset('episode_ends', data=episode_ends)

    print(f"✅ Success! Dataset created at {OUTPUT_ZARR}")
    print(f"   Total Frames: {total_steps}")
    print(f"   Image Shape: {combined_imgs.shape}")
    print(f"   Action Shape: {combined_actions.shape}")

if __name__ == "__main__":
    create_dataset()