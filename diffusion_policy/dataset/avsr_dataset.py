"""
Dataset for avsr_teleop raw data format.

Directory layout written by avsr_teleop/scripts/teleop_node.py + ros_recorder.py:

  <dataset_path>/
    episodes/
      <i>/
        trajectory_<unix_ms>.json   ← waypoints at record_hz (default 10 Hz)
        rgb_frames/
          cam1_<ts_ms>.png          ← wrist camera at 30 Hz
          cam2_<ts_ms>.png          ← external camera (optional)

Each trajectory JSON has the structure:
  {
    "metadata": { ... },
    "waypoints": [
      {
        "timestamp":    float,       # time.time() seconds
        "position":     [x, y, z],  # target EE position
        "orientation":  [x, y, z, w], # target EE quaternion
        "gripper":      bool,        # target gripper state
        "proc_pos":     [x, y, z],  # actual EE position
        "proc_quat":    [x, y, z, w], # actual EE quaternion
        "proc_gripper": bool,        # actual gripper state
        "joint_pos":    [j1..j6]    # actual joint positions (rad)
      }, ...
    ]
  }

Action space  (4D): actual EE position proc_pos (3) + commanded gripper state (1 float).
  Using proc_pos (FK from joint encoders) rather than the VR-commanded `position`
  avoids injecting VR tracking noise into the action labels. Gripper uses the
  commanded `gripper` field (not proc_gripper, which was not recorded correctly).

Agent_pos (7D): actual joint angles joint_pos (6) + commanded gripper state (1 float).
  Joint angles uniquely determine arm configuration; EE position alone is ambiguous
  (multiple joint configs can reach the same Cartesian pose).

Image: wrist camera resized to 240x320 (H x W), normalised to [0, 1]
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer

# Target image size expected by MultiImageObsEncoder (H, W)
_IMG_H, _IMG_W = 240, 320


def _load_all_episodes(dataset_path: str) -> List[dict]:
    """Scan dataset_path/episodes/ and return a list of episode dicts.

    Each dict:
      actions   : np.float32 (N, 4)  — target [x, y, z, gripper]
      agent_pos : np.float32 (N, 4)  — actual [x, y, z, gripper]
      img_paths : list[str]          — absolute path to wrist cam frame per step
    """
    root = Path(dataset_path).expanduser()
    ep_dirs = sorted(
        [d for d in (root / "episodes").iterdir() if d.is_dir()],
        key=lambda p: int(p.name),
    )

    episodes = []
    for ep_dir in ep_dirs:
        traj_files = sorted(ep_dir.glob("trajectory_*.json"))
        if not traj_files:
            continue

        with open(traj_files[-1]) as f:
            traj = json.load(f)
        waypoints = traj.get("waypoints", [])
        if not waypoints:
            continue

        rgb_dir = ep_dir / "rgb_frames"
        cam1_files = sorted(rgb_dir.glob("cam1_*.png"))
        if not cam1_files:
            continue

        # Build timestamp array (ms) for camera frames
        cam1_ts = np.array([int(p.stem.split("_")[1]) for p in cam1_files])

        N = len(waypoints)
        actions = np.zeros((N, 4), dtype=np.float32)   # proc_pos (3) + proc_gripper (1)
        agent_pos = np.zeros((N, 7), dtype=np.float32)  # joint_pos (6) + proc_gripper (1)
        img_paths: List[str] = []

        for i, wp in enumerate(waypoints):
            # Action: actual EE pos (FK) + commanded gripper state
            actions[i, :3] = wp["proc_pos"]
            actions[i, 3] = float(wp["gripper"])
            # State: joint angles (unique config) + gripper
            agent_pos[i, :6] = wp["joint_pos"]
            agent_pos[i, 6] = float(wp["gripper"])

            # Find nearest camera frame by timestamp
            ts_ms = int(wp["timestamp"] * 1000)
            closest_idx = int(np.argmin(np.abs(cam1_ts - ts_ms)))
            img_paths.append(str(cam1_files[closest_idx]))

        episodes.append(
            {"actions": actions, "agent_pos": agent_pos, "img_paths": img_paths}
        )

    return episodes


def _load_image(path: str) -> np.ndarray:
    """Load wrist cam frame → float32 (3, H, W) in [0, 1], resized to _IMG_H x _IMG_W."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((3, _IMG_H, _IMG_W), dtype=np.float32)
    if img.shape[:2] != (_IMG_H, _IMG_W):
        img = cv2.resize(img, (_IMG_W, _IMG_H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)  # (3, H, W)


class AvSRDataset(BaseImageDataset):
    """Single-camera dataset for data collected with avsr_teleop.

    Samples windows of `horizon` consecutive steps from each episode.
    With pad_before / pad_after, the first / last episode steps are repeated
    at the boundaries (identical to SequenceSampler boundary behaviour).

    Each sample:
      obs/camera_0  : float32 (horizon, 3, H, W)
      obs/agent_pos : float32 (horizon, 7)   joint_pos (6) + proc_gripper (1)
      action        : float32 (horizon, 4)   proc_pos  (3) + proc_gripper (1)
    """

    def __init__(
        self,
        dataset_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.05,
        max_train_episodes: Optional[int] = None,
        shape_meta: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        print(f"[AvSRDataset] Scanning {dataset_path} …")
        all_episodes = _load_all_episodes(dataset_path)
        n = len(all_episodes)
        print(f"[AvSRDataset] Found {n} valid episodes.")

        # Reproducible train / val split
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n).tolist()
        n_val = max(1, int(round(n * val_ratio)))
        val_set = set(perm[:n_val])
        train_list = [i for i in perm[n_val:] if i not in val_set]
        if max_train_episodes is not None:
            train_list = train_list[:max_train_episodes]

        self._all_episodes = all_episodes
        self._train_list = train_list
        self._val_list = list(val_set)
        self._index = self._build_flat_index(train_list)

    # ── Index helpers ─────────────────────────────────────────────────────────

    def _build_flat_index(self, ep_indices: List[int]) -> List[tuple]:
        """One entry per episode step; window is centred on `step` with padding."""
        index = []
        for ep_i in ep_indices:
            N = len(self._all_episodes[ep_i]["actions"])
            for step in range(N):
                index.append((ep_i, step))
        return index

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_i, center = self._index[idx]
        ep = self._all_episodes[ep_i]
        N = len(ep["actions"])

        # Window: [center - pad_before, center - pad_before + horizon)
        start = center - self.pad_before
        actions = []
        agent_pos = []
        images = []

        for t in range(start, start + self.horizon):
            t_c = max(0, min(t, N - 1))
            actions.append(ep["actions"][t_c])
            agent_pos.append(ep["agent_pos"][t_c])
            images.append(_load_image(ep["img_paths"][t_c]))

        data = {
            "obs": {
                "camera_0": np.stack(images),       # (T, 3, H, W)
                "agent_pos": np.stack(agent_pos),   # (T, 4)
            },
            "action": np.stack(actions),            # (T, 4)
        }
        return dict_apply(data, torch.from_numpy)

    def get_validation_dataset(self) -> "AvSRDataset":
        val_set = copy.copy(self)
        val_set._train_list = self._val_list
        val_set._val_list = self._train_list
        val_set._index = self._build_flat_index(self._val_list)
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        ep_list = [self._all_episodes[i] for i in self._train_list]
        all_actions = np.concatenate([e["actions"] for e in ep_list], axis=0)
        all_pos = np.concatenate([e["agent_pos"] for e in ep_list], axis=0)

        normalizer.fit(
            data={"action": all_actions, "agent_pos": all_pos},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        normalizer["camera_0"] = get_image_range_normalizer()
        return normalizer
