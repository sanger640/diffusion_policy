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
        "timestamp":    float,         # time.time() seconds
        "position":     [x, y, z],     # target EE position (VR commanded)
        "orientation":  [x, y, z, w],  # target EE quaternion (VR commanded)
        "gripper":      bool,          # commanded gripper state  ← USE THIS
        "proc_pos":     [x, y, z],     # actual EE position (FK from joint encoders)
        "proc_quat":    [x, y, z, w],  # actual EE quaternion (FK from joint encoders)
        "proc_gripper": bool,          # PREVIOUS step's actual gripper state — do not use
        "joint_pos":    [j1..j6],      # actual joint positions (rad)
        "ee_twist":     [vx,vy,vz,wx,wy,wz]  # actual EE spatial velocity (optional; zeros if not recorded)
      }, ...
    ]
  }

Gripper field:  always use "gripper" (commanded current state), never "proc_gripper"
                (proc_gripper lags by one step and was not reliably recorded).

Action space:
  Position-only (action_mode='ee', action_dim=4):
    proc_pos (3) + gripper (1)
  With rot6d orientation (action_mode='ee_ori' / include_orientation=True, action_dim=10):
    proc_pos (3) + rot6d_from_proc_quat (6) + gripper (1)
  With raw quaternion orientation (action_mode='ee_quat', action_dim=8):
    proc_pos (3) + proc_quat (4) + gripper (1)
  Joint-space (action_mode='joints', action_dim=7):
    joint_pos (6) + gripper (1)

  proc_pos / proc_quat come from FK (joint encoder readback) rather than the
  spacemouse-commanded position/orientation to avoid injecting controller/VR
  tracking noise into labels.

Agent_pos: joint_pos (6 rad) + gripper (1 float) = 7D by default.
  With include_velocity=True, ee_twist (6D: vx,vy,vz,wx,wy,wz) is appended → 13D.
  Joint angles uniquely determine arm configuration; EE position alone is ambiguous.

Image: wrist camera resized to 240x320 (H x W), normalised to [0, 1]
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer

# Target image size expected by MultiImageObsEncoder (H, W)
_IMG_H, _IMG_W = 240, 320


def _resample_waypoints(waypoints: list, hz: float = 10.0) -> list:
    """Resample variable-rate waypoints to a fixed rate via linear interpolation.

    Eliminates timing gaps from VR teleop (observed max gap: 749ms vs target 100ms).
    Gripper uses nearest-neighbour to preserve binary transitions.
    Quaternions are linearly interpolated per-component then renormalised.
    """
    if len(waypoints) < 2:
        return waypoints
    ts = np.array([w["timestamp"] for w in waypoints])
    t_new = np.arange(ts[0], ts[-1], 1.0 / hz)
    if len(t_new) < 4:
        return waypoints

    joints_raw   = np.array([w["joint_pos"]  for w in waypoints])   # (N, 6)
    proc_pos_raw = np.array([w["proc_pos"]   for w in waypoints])   # (N, 3)
    proc_quat_raw= np.array([w["proc_quat"]  for w in waypoints])   # (N, 4)
    twist_raw    = np.array([w.get("ee_twist", [0.0] * 6) for w in waypoints])  # (N, 6)
    grip_raw     = np.array([float(w["gripper"]) for w in waypoints])

    joints_new   = np.stack([np.interp(t_new, ts, joints_raw[:, j])   for j in range(6)], axis=1)
    proc_pos_new = np.stack([np.interp(t_new, ts, proc_pos_raw[:, k]) for k in range(3)], axis=1)
    proc_quat_new= np.stack([np.interp(t_new, ts, proc_quat_raw[:, k])for k in range(4)], axis=1)
    proc_quat_new /= np.maximum(np.linalg.norm(proc_quat_new, axis=1, keepdims=True), 1e-6)
    twist_new    = np.stack([np.interp(t_new, ts, twist_raw[:, k])    for k in range(6)], axis=1)

    # nearest-neighbour for gripper to keep binary transitions sharp
    grip_idx = np.argmin(np.abs(t_new[:, None] - ts[None, :]), axis=1)
    grip_new = grip_raw[grip_idx] > 0.5

    return [
        {
            "timestamp": float(t_new[i]),
            "joint_pos": joints_new[i].tolist(),
            "proc_pos":  proc_pos_new[i].tolist(),
            "proc_quat": proc_quat_new[i].tolist(),
            "ee_twist":  twist_new[i].tolist(),
            "gripper":   bool(grip_new[i]),
        }
        for i in range(len(t_new))
    ]


def _smooth_array(arr: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing per column. Reduces VR controller jitter
    (~20% direction-reversal rate) without distorting the underlying motion."""
    wl = min(window, (len(arr) // 2) * 2 - 1)
    wl = max(wl, poly + 2)
    if len(arr) < wl:
        return arr
    return np.stack([savgol_filter(arr[:, j], wl, poly)
                     for j in range(arr.shape[1])], axis=1)


def _quat_to_rot6d(quats: np.ndarray) -> np.ndarray:
    """Convert (N, 4) [x,y,z,w] quaternions → (N, 6) rot6d (first two columns of rotation matrix).

    Rot6d is a continuous representation that avoids the antipodal symmetry of quaternions,
    which improves diffusion policy training on orientation-controlled tasks.
    """
    matrices = R.from_quat(quats).as_matrix()                                   # (N, 3, 3)
    return np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1)       # (N, 6)


def _load_all_episodes(dataset_path: str, include_orientation: bool = False,
                       action_mode: str = 'ee',
                       resample_hz: float = 10.0,
                       smooth_window: int = 7,
                       include_velocity: bool = False) -> List[dict]:
    """Scan dataset_path/episodes/ and return a list of episode dicts.

    Each dict:
      actions   : np.float32 (N, 4, 7, 8 or 10) — see module docstring
      agent_pos : np.float32 (N, 7 or 13)         — joint_pos(6) + gripper(1) [+ ee_twist(6)]
      images    : np.float32 (N, 3, H, W)         — wrist cam frames pre-loaded into RAM

    action_mode:
      'ee'      — proc_pos(3) + gripper(1) = 4D
      'ee_ori'  — proc_pos(3) + rot6d(6) + gripper(1) = 10D  (same as include_orientation=True)
      'ee_quat' — proc_pos(3) + proc_quat(4) + gripper(1) = 8D
      'joints'  — joint_pos(6) + gripper(1) = 7D

    include_velocity:
      If True, append ee_twist (6D: vx,vy,vz,wx,wy,wz) to agent_pos, giving 13D instead of 7D.

    resample_hz:
      Resample waypoints to this fixed rate before building arrays.
      Eliminates timing gaps (observed max 749ms vs target 100ms in fine_pick_v2).
      Set to 0 to disable.

    smooth_window:
      Savitzky-Golay window length for joint/EE position smoothing.
      Reduces VR controller jitter (~20% direction-reversal rate).
      Set to 0 to disable.
    """
    if action_mode == 'joints':
        action_dim = 7
    elif action_mode == 'ee_ori' or include_orientation:
        action_dim = 10
    elif action_mode == 'ee_quat':
        action_dim = 8
    else:
        action_dim = 4

    agent_pos_dim = 13 if include_velocity else 7

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

        # ── Filter: drop episodes too corrupted for reliable interpolation ────
        # Filtering alone can't fix the dataset (96% of episodes have some gap),
        # but episodes with extreme gaps or high gap density are unfixable by
        # resampling — linear interpolation over 500ms+ fabricates unknown motion.
        if len(waypoints) >= 2 and resample_hz > 0:
            _dts = np.diff([w["timestamp"] for w in waypoints])
            _max_gap   = float(_dts.max())
            _gap_frac  = float(np.sum(_dts > 0.20) / len(_dts))
            if _max_gap > 0.50 or _gap_frac > 0.03:
                continue

        # ── Fix 1: resample to fixed rate (removes timing gaps) ───────────────
        if resample_hz > 0:
            waypoints = _resample_waypoints(waypoints, hz=resample_hz)
        if not waypoints:
            continue

        rgb_dir = ep_dir / "rgb_frames"
        cam1_files = sorted(rgb_dir.glob("cam1_*.png"))
        if not cam1_files:
            continue

        # Build timestamp array (ms) for camera frames
        cam1_ts = np.array([int(p.stem.split("_")[1]) for p in cam1_files])

        N = len(waypoints)
        actions   = np.zeros((N, action_dim), dtype=np.float32)
        agent_pos = np.zeros((N, agent_pos_dim), dtype=np.float32)
        images    = np.zeros((N, 3, _IMG_H, _IMG_W), dtype=np.float32)

        proc_pos_all  = np.zeros((N, 3), dtype=np.float32)
        proc_quat_all = np.zeros((N, 4), dtype=np.float32)
        grip_all      = np.zeros((N, 1), dtype=np.float32)

        for i, wp in enumerate(waypoints):
            proc_pos_all[i]  = wp["proc_pos"]
            proc_quat_all[i] = wp["proc_quat"]   # [x, y, z, w]
            grip_all[i, 0]   = float(wp["gripper"])   # commanded state — not proc_gripper

            agent_pos[i, :6] = wp["joint_pos"]
            agent_pos[i, 6]  = float(wp["gripper"])
            if include_velocity:
                agent_pos[i, 7:13] = wp.get("ee_twist", [0.0] * 6)

            # Find nearest camera frame by timestamp and load into RAM
            ts_ms = int(wp["timestamp"] * 1000)
            closest_idx = int(np.argmin(np.abs(cam1_ts - ts_ms)))
            images[i] = _load_image(str(cam1_files[closest_idx]))

        # ── Fix 2: smooth joint/EE positions (reduces direction reversals) ────
        if smooth_window > 0:
            agent_pos[:, :6] = _smooth_array(agent_pos[:, :6], window=smooth_window)
            if action_mode in ('ee', 'ee_ori', 'ee_quat') or include_orientation:
                proc_pos_all = _smooth_array(proc_pos_all, window=smooth_window)

        if action_mode == 'joints':
            joint_all = agent_pos[:, :6]
            actions = np.concatenate([joint_all, grip_all], axis=1)            # (N, 7)
        elif action_mode == 'ee_ori' or include_orientation:
            rot6d = _quat_to_rot6d(proc_quat_all)          # (N, 6)
            actions = np.concatenate([proc_pos_all, rot6d, grip_all], axis=1)  # (N, 10)
        elif action_mode == 'ee_quat':
            actions = np.concatenate([proc_pos_all, proc_quat_all, grip_all], axis=1)  # (N, 8)
        else:
            actions = np.concatenate([proc_pos_all, grip_all], axis=1)         # (N, 4)

        episodes.append(
            {"actions": actions, "agent_pos": agent_pos, "images": images}
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

    Each sample (action_mode='ee'):
      obs/camera_0  : float32 (horizon, 3, H, W)
      obs/agent_pos : float32 (horizon, 7 or 13)   joint_pos(6) + gripper(1) [+ ee_twist(6)]
      action        : float32 (horizon, 4)         proc_pos(3)  + gripper(1)

    Each sample (action_mode='ee_ori' / include_orientation=True):
      obs/camera_0  : float32 (horizon, 3, H, W)
      obs/agent_pos : float32 (horizon, 7 or 13)   joint_pos(6) + gripper(1) [+ ee_twist(6)]
      action        : float32 (horizon, 10)        proc_pos(3) + rot6d(6) + gripper(1)

    Each sample (action_mode='ee_quat'):
      obs/camera_0  : float32 (horizon, 3, H, W)
      obs/agent_pos : float32 (horizon, 7 or 13)   joint_pos(6) + gripper(1) [+ ee_twist(6)]
      action        : float32 (horizon, 8)         proc_pos(3) + proc_quat(4) + gripper(1)

    include_velocity=True appends ee_twist (6D) to agent_pos, giving 13D instead of 7D.
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
        include_orientation: bool = False,
        action_mode: str = 'ee',
        resample_hz: float = 10.0,
        smooth_window: int = 7,
        include_velocity: bool = False,
        shape_meta: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.include_orientation = include_orientation
        self.action_mode = action_mode
        self.include_velocity = include_velocity

        mode_str = action_mode if action_mode != 'ee' else ('ee_ori' if include_orientation else 'ee')
        print(f"[AvSRDataset] Scanning {dataset_path} (pre-loading into RAM) …  "
              f"action_mode={mode_str}  resample_hz={resample_hz}  smooth_window={smooth_window}  "
              f"include_velocity={include_velocity}")
        all_episodes = _load_all_episodes(dataset_path,
                                          include_orientation=include_orientation,
                                          action_mode=action_mode,
                                          resample_hz=resample_hz,
                                          smooth_window=smooth_window,
                                          include_velocity=include_velocity)
        n = len(all_episodes)
        print(f"[AvSRDataset] Loaded {n} episodes into RAM.")

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
            images.append(ep["images"][t_c])

        data = {
            "obs": {
                "camera_0": np.stack(images),       # (T, 3, H, W)
                "agent_pos": np.stack(agent_pos),   # (T, 7) or (T, 13)
            },
            "action": np.stack(actions),            # (T, 4), (T, 7), (T, 8) or (T, 10)
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

        normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_fit(
            all_pos, last_n_dims=1, mode=mode, **kwargs)
        normalizer["action"] = self._fit_action_normalizer(all_actions, mode=mode, **kwargs)
        normalizer["camera_0"] = get_image_range_normalizer()
        return normalizer

    def _fit_action_normalizer(self, all_actions: np.ndarray, mode: str = "limits",
                                **kwargs) -> SingleFieldLinearNormalizer:
        """Rotation dims (rot6d or raw quaternion) are left identity-normalized
        (scale=1, offset=0) instead of independently min/max-scaled per component.

        Per-dimension min/max scaling is fine for position and gripper (each is an
        independent, unconstrained scalar), but rotation representations encode a
        constrained manifold (unit quaternion / orthonormal rot6d columns) — scaling
        each component by its own observed range breaks the isometry between
        normalized-space distance and true rotational distance, which the diffusion
        loss then optimizes against. This mirrors diffusion_policy's own
        robomimic_abs_action_only_normalizer_from_stat (normalize_util.py), which
        applies identity scale/offset to the rotation block of real/sim absolute
        end-effector actions and only min/max-fits position.
        """
        mode_str = self.action_mode if self.action_mode != 'ee' else (
            'ee_ori' if self.include_orientation else 'ee')
        rot_slice = {'ee_ori': slice(3, 9), 'ee_quat': slice(3, 7)}.get(mode_str)

        if rot_slice is None:
            return SingleFieldLinearNormalizer.create_fit(
                all_actions, last_n_dims=1, mode=mode, **kwargs)

        D = all_actions.shape[-1]
        flat = all_actions.reshape(-1, D).astype(np.float32)
        other_idx = np.array(
            [i for i in range(D) if not (rot_slice.start <= i < rot_slice.stop)])

        other_normalizer = SingleFieldLinearNormalizer.create_fit(
            flat[:, other_idx], last_n_dims=1, mode=mode, **kwargs)
        other_p = other_normalizer.params_dict
        rot = flat[:, rot_slice]

        scale = np.zeros(D, dtype=np.float32)
        offset = np.zeros(D, dtype=np.float32)
        stats = {k: np.zeros(D, dtype=np.float32) for k in ('min', 'max', 'mean', 'std')}

        scale[other_idx] = other_p['scale'].detach().cpu().numpy()
        offset[other_idx] = other_p['offset'].detach().cpu().numpy()
        for k in stats:
            stats[k][other_idx] = other_p['input_stats'][k].detach().cpu().numpy()

        scale[rot_slice] = 1.0
        offset[rot_slice] = 0.0
        stats['min'][rot_slice] = rot.min(axis=0)
        stats['max'][rot_slice] = rot.max(axis=0)
        stats['mean'][rot_slice] = rot.mean(axis=0)
        stats['std'][rot_slice] = rot.std(axis=0)

        return SingleFieldLinearNormalizer.create_manual(
            scale=scale, offset=offset, input_stats_dict=stats)
