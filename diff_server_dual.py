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
    config_name="train_franka_dual_cup"
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
        h, w = 240, 320 # Or read from message['shape']
        # B. Unpack Data
        # ADD .copy() TO BOTH LINES
        c1_np = np.frombuffer(message['cam1'], dtype=np.uint8).reshape(1, 2, 3, h, w).copy()
        c2_np = np.frombuffer(message['cam2'], dtype=np.uint8).reshape(1, 2, 3, h, w).copy()

        s_np = np.frombuffer(message['state'], dtype=np.float32).reshape(1, 2, 4).copy()
        # state_np = np.frombuffer(message['state'], dtype=np.float32).reshape(1, 2, 4).copy()
        # C. Preprocess
        # Convert 0-255 -> 0.0-1.0
        c1_t = torch.from_numpy(c1_np).float().to(device) / 255.0
        c2_t = torch.from_numpy(c2_np).float().to(device) / 255.0
        s_t  = torch.from_numpy(s_np).float().to(device)

        obs_dict = {
            'camera_1': c1_t,
            'camera_2': c2_t,
            'agent_pos': s_t
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