import zmq
import torch
import numpy as np
import hydra
import dill
import pathlib
import time
from omegaconf import OmegaConf
 
# --- CONFIG ---
CHECKPOINT_PATH = "data/outputs/2025.12.03/latest.ckpt"
PORT = 5555
# --------------
 
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('diffusion_policy', 'config')),
    config_name="train_franka_cup"
)
def main(cfg: OmegaConf):
    # 1. Load Model
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    payload = torch.load(open(CHECKPOINT_PATH, 'rb'), pickle_module=dill)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device('cuda')
    policy.to(device)
    policy.eval()
    policy.num_inference_steps = 10

    print("✅ Model Loaded. Starting Network Server...")
 
    # 2. Setup ZMQ Server
    context = zmq.Context()
    socket = context.socket(zmq.REP) # Reply socket
    socket.bind(f"tcp://*:{PORT}")
    print(f"Listening on port {PORT}...")
 
    # 3. Server Loop
    while True:
        # A. Wait for Request (Blocking)
        # We expect a dictionary: {'img': bytes, 'state': bytes, 'shape': tuple}
        message = socket.recv_pyobj()
        # print("recived")
        start_time = time.time()
        # B. Unpack Data
        # Image comes as flat buffer, reshape it
        img_h, img_w = 240, 320 # Or read from message['shape']
        # B. Unpack Data
        # ADD .copy() TO BOTH LINES
        img_np = np.frombuffer(message['img'], dtype=np.uint8).reshape(1, 2, 3, img_h, img_w).copy()
        state_np = np.frombuffer(message['state'], dtype=np.float32).reshape(1, 2, 4).copy()
        # C. Preprocess
        # Convert 0-255 -> 0.0-1.0
        img_tensor = torch.from_numpy(img_np).float().to(device) / 255.0
        state_tensor = torch.from_numpy(state_np).float().to(device)
        obs_dict = {
            'agent_view_image': img_tensor,
            'agent_pos': state_tensor
        }
 
        # D. Inference
        with torch.no_grad():
            result = policy.predict_action(obs_dict)
        # E. Pack Action
        # Take the first 8 steps of the prediction (Receding Horizon)
        action_pred = result['action'][0].cpu().numpy() # (16, 10)
        action_chunk = action_pred[:8] # Send 8 steps back
        # F. Send Reply
        socket.send_pyobj({
            'action': action_chunk,
            'inference_time': time.time() - start_time
        })
        # print(f"Processed request in {time.time() - start_time:.4f}s")
 
if __name__ == "__main__":
    main()