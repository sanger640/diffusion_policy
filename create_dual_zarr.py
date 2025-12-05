import os
import json
import numpy as np
import cv2
import zarr
from tqdm import tqdm
from glob import glob
from scipy.spatial.transform import Rotation as R
from numcodecs import Blosc

# --- CONFIGURATION ---
DATA_ROOT = "/media/corey/New Volume/diffusion_policy/tasks/pick_place/dual_cam/episodes"  # Your data folder
OUTPUT_ZARR = "data/paper_implementation.zarr"
RESIZE_W, RESIZE_H = 320, 240 # Resolution matches your training config

def quat_to_rot6d(quat_list):
    """
    Converts list of quaternions [x,y,z,w] to [r1...r6]
    """
    r = R.from_quat(quat_list)
    matrices = r.as_matrix() # (N, 3, 3)
    rot6d = matrices[:, :, :2].reshape(-1, 6)
    return rot6d

def create_dataset():
    if os.path.exists(OUTPUT_ZARR):
        print(f"Dataset {OUTPUT_ZARR} already exists. Please delete it first.")
        return

    # 1. Collect Episodes
    episode_dirs = sorted(glob(os.path.join(DATA_ROOT, "*")))
    
    all_cam1_imgs = []
    all_cam2_imgs = []
    all_states = []
    all_actions = []
    episode_ends = []

    print(f"Processing {len(episode_dirs)} episodes (Multi-Camera)...")

    for ep_dir in tqdm(episode_dirs):
        # Load Trajectory JSON
        json_files = glob(os.path.join(ep_dir, "*.json"))
        if not json_files: continue
        
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        waypoints = data['waypoints']
        
        # Load Images
        rgb_dir = os.path.join(ep_dir, "rgb_frames")
        
        # We rely on the fact that your recorder saves cam1 and cam2 with IDENTICAL timestamps
        # So we only need to sort based on one of them to build the index
        cam1_files = sorted(glob(os.path.join(rgb_dir, "cam1_*.png")))
        
        # Parse timestamps from filenames (format: cam1_1701234567890.png)
        img_timestamps = []
        file_map = {} # Map timestamp -> filename suffix
        
        for f in cam1_files:
            fname = os.path.basename(f)
            # Extract the number part: "cam1_12345.png" -> "12345"
            ts_str = fname.replace('cam1_', '').replace('.png', '')
            try:
                # Recorder saved as ms (int), robot uses seconds (float)
                ts_float = float(ts_str) / 1000.0
                img_timestamps.append(ts_float)
                file_map[ts_float] = ts_str
            except ValueError:
                continue
                
        img_timestamps = np.array(img_timestamps)

        # --- Process Trajectory ---
        positions = []
        quats = []
        grippers = []
        
        # We store the robot timestamps to sync with images
        robot_timestamps = []

        for wp in waypoints:
            positions.append(wp['position'])
            quats.append(wp['orientation'])
            grippers.append([1.0] if wp['gripper'] else [-1.0])
            robot_timestamps.append(wp['timestamp'])

        # Convert Rotation to 6D (We save full 10D state to disk, filter later if needed)
        rot6d = quat_to_rot6d(quats)
        
        states_np = np.concatenate([
            np.array(positions), 
            rot6d, 
            np.array(grippers)
        ], axis=1).astype(np.float32)

        # --- Process Images (Sync) ---
        ep_cam1 = []
        ep_cam2 = []
        
        for t_robot in robot_timestamps:
            # Find closest image timestamp
            if len(img_timestamps) == 0:
                print(f"Warning: No images found for {ep_dir}")
                break
                
            diffs = np.abs(img_timestamps - t_robot)
            idx = np.argmin(diffs)
            best_ts = img_timestamps[idx]
            suffix = file_map[best_ts]
            
            # Construct paths for BOTH cameras using the matched suffix
            path1 = os.path.join(rgb_dir, f"cam1_{suffix}.png")
            path2 = os.path.join(rgb_dir, f"cam2_{suffix}.png")
            
            # Load & Resize Cam 1
            if os.path.exists(path1):
                img1 = cv2.imread(path1)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img1 = cv2.resize(img1, (RESIZE_W, RESIZE_H))
                ep_cam1.append(np.transpose(img1, (2, 0, 1)))
            else:
                # Fallback if missing (should not happen with bundled queue)
                ep_cam1.append(np.zeros((3, RESIZE_H, RESIZE_W), dtype=np.uint8))

            # Load & Resize Cam 2
            if os.path.exists(path2):
                img2 = cv2.imread(path2)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                img2 = cv2.resize(img2, (RESIZE_W, RESIZE_H))
                ep_cam2.append(np.transpose(img2, (2, 0, 1)))
            else:
                ep_cam2.append(np.zeros((3, RESIZE_H, RESIZE_W), dtype=np.uint8))

        # Append to episode lists
        all_cam1_imgs.append(np.array(ep_cam1))
        all_cam2_imgs.append(np.array(ep_cam2))
        all_states.append(states_np)
        all_actions.append(states_np) # Action = Next State (Auto-regressive)
        episode_ends.append(len(states_np))

    # --- Write to Zarr ---
    print("Concatenating data...")
    combined_cam1 = np.concatenate(all_cam1_imgs, axis=0)
    combined_cam2 = np.concatenate(all_cam2_imgs, axis=0)
    combined_states = np.concatenate(all_states, axis=0)
    combined_actions = np.concatenate(all_actions, axis=0)
    episode_ends = np.cumsum(episode_ends)

    print(f"Writing to {OUTPUT_ZARR}...")
    root = zarr.open(OUTPUT_ZARR, mode='w')
    
    # Structure
    data_group = root.create_group('data')
    obs_group = data_group.create_group('obs')
    meta_group = root.create_group('meta')
    
    # Metadata
    total_steps = int(episode_ends[-1])
    root.attrs["total"] = total_steps
    root.attrs["env_args"] = json.dumps({"env_name": "FrankaRealMultiCam", "env_version": 1.0})
    root.attrs["layout"] = "robomimic"

    compressor = Blosc(cname='zstd', clevel=3, shuffle=1)
    
    # Datasets
    # 1. Camera 1
    obs_group.create_dataset('camera_1', data=combined_cam1, 
                             chunks=(100, 3, RESIZE_H, RESIZE_W), compressor=compressor)
    
    # 2. Camera 2
    obs_group.create_dataset('camera_2', data=combined_cam2, 
                             chunks=(100, 3, RESIZE_H, RESIZE_W), compressor=compressor)
    
    # 3. State
    obs_group.create_dataset('agent_pos', data=combined_states, 
                             chunks=(100, 10), compressor=compressor)
    
    # 4. Action
    data_group.create_dataset('action', data=combined_actions, 
                              chunks=(100, 10), compressor=compressor)
    
    # 5. Meta
    meta_group.create_dataset('episode_ends', data=episode_ends)

    print(f"✅ Success!")
    print(f"   Total Frames: {total_steps}")
    print(f"   Cam 1 Shape: {combined_cam1.shape}")
    print(f"   Cam 2 Shape: {combined_cam2.shape}")

if __name__ == "__main__":
    create_dataset()