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
DATA_ROOT = "tasks/jenga_mujoco/episodes"
OUTPUT_ZARR = "data/jenga_muj_imp.zarr"
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
    all_states = []    # Will store proc (proprioception)
    all_actions = []   # Will store targets (commands)
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
        cam1_files = sorted(glob(os.path.join(rgb_dir, "cam1_*.png")))
        
        img_timestamps = []
        file_map = {} 
        
        for f in cam1_files:
            fname = os.path.basename(f)
            ts_str = fname.replace('cam1_', '').replace('.png', '')
            try:
                ts_float = float(ts_str) / 1000.0
                img_timestamps.append(ts_float)
                file_map[ts_float] = ts_str
            except ValueError:
                continue
                
        img_timestamps = np.array(img_timestamps)

        # --- Process Trajectory ---
        cmd_positions, cmd_quats, cmd_grippers = [], [], []
        proc_positions, proc_quats, proc_grippers = [], [], []
        robot_timestamps = []

        for wp in waypoints:
            # Action (Commanded Targets)
            cmd_positions.append(wp['position'])
            cmd_quats.append(wp['orientation'])
            cmd_grippers.append([1.0] if wp['gripper'] else [-1.0])
            
            # State (Actual Proprioception)
            proc_positions.append(wp['proc_pos'])
            proc_quats.append(wp['proc_quat'])
            proc_grippers.append([1.0] if wp['proc_gripper'] else [-1.0])
            
            robot_timestamps.append(wp['timestamp'])

        # Convert Rotations to 6D
        cmd_rot6d = quat_to_rot6d(cmd_quats)
        proc_rot6d = quat_to_rot6d(proc_quats)
        
        # Build 10D Arrays (XYZ + Rot6D + Grip)
        actions_np = np.concatenate([
            np.array(cmd_positions), 
            cmd_rot6d, 
            np.array(cmd_grippers)
        ], axis=1).astype(np.float32)

        states_np = np.concatenate([
            np.array(proc_positions), 
            proc_rot6d, 
            np.array(proc_grippers)
        ], axis=1).astype(np.float32)

        # --- Process Images (Sync) ---
        ep_cam1 = []
        ep_cam2 = []
        
        for t_robot in robot_timestamps:
            if len(img_timestamps) == 0:
                print(f"Warning: No images found for {ep_dir}")
                break
                
            diffs = np.abs(img_timestamps - t_robot)
            idx = np.argmin(diffs)
            best_ts = img_timestamps[idx]
            suffix = file_map[best_ts]
            
            path1 = os.path.join(rgb_dir, f"cam1_{suffix}.png")
            path2 = os.path.join(rgb_dir, f"cam2_{suffix}.png")
            
            if os.path.exists(path1):
                img1 = cv2.imread(path1)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img1 = cv2.resize(img1, (RESIZE_W, RESIZE_H))
                ep_cam1.append(np.transpose(img1, (2, 0, 1)))
            else:
                ep_cam1.append(np.zeros((3, RESIZE_H, RESIZE_W), dtype=np.uint8))

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
        
        # MAP STATE TO PROC, ACTION TO CMD
        all_states.append(states_np)
        all_actions.append(actions_np) 
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
    
    data_group = root.create_group('data')
    obs_group = data_group.create_group('obs')
    meta_group = root.create_group('meta')
    
    total_steps = int(episode_ends[-1])
    root.attrs["total"] = total_steps
    root.attrs["env_args"] = json.dumps({"env_name": "FrankaRealMultiCam", "env_version": 1.0})
    root.attrs["layout"] = "robomimic"

    compressor = Blosc(cname='zstd', clevel=3, shuffle=1)
    
    # Datasets
    obs_group.create_dataset('camera_1', data=combined_cam1, chunks=(100, 3, RESIZE_H, RESIZE_W), compressor=compressor)
    obs_group.create_dataset('camera_2', data=combined_cam2, chunks=(100, 3, RESIZE_H, RESIZE_W), compressor=compressor)
    
    # State = Proprioception
    obs_group.create_dataset('agent_pos', data=combined_states, chunks=(100, 10), compressor=compressor)
    # Action = Targets
    data_group.create_dataset('action', data=combined_actions, chunks=(100, 10), compressor=compressor)
    
    meta_group.create_dataset('episode_ends', data=episode_ends)

    print(f"✅ Success!")
    print(f"   Total Frames: {total_steps}")

if __name__ == "__main__":
    create_dataset()