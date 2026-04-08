"""
extracts egomotion and camera from NVIDIA physical AI AV dataset
"""
import physical_ai_av
import numpy as np
import torch
import scipy.spatial.transform as spt
from einops import rearrange

camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }


def get_egomotion_for_curr_t(e_m, clip_id, t0_us, config):
    assert t0_us > config.num_history_steps * config.time_step * 1_000_000, (
            "t0_us must be greater than the history time range"
        )

    # Compute timestamps for trajectory sampling
    # History: [..., t0-0.2s, t0-0.1s, t0] (num_history_steps points ending at t0)
    # Future: [t0+0.1s, t0+0.2s, ..., t0+6.4s] (num_future_steps points after t0)
    
    history_offsets_us = np.arange(
        -(config.num_history_steps - 1) * config.time_step * 1_000_000,
        config.time_step * 1_000_000 / 2,
        config.time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    # Get egomotion at history and future timestamps
    ego_history = e_m(history_timestamps)
    ego_history_xyz = ego_history.pose.translation  # (num_history_steps, 3)
    ego_history_quat = ego_history.pose.rotation.as_quat()  # (num_history_steps, 4)


    future_offsets_us = np.arange(
        config.time_step * 1_000_000,
        (config.num_future_steps + 0.5) * config.time_step * 1_000_000,
        config.time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    ego_future = e_m(future_timestamps)
    ego_future_xyz = ego_future.pose.translation  # (num_future_steps, 3)
    ego_future_quat = ego_future.pose.rotation.as_quat()  # (num_future_steps, 4)

    # Transform to local frame (relative to t0 pose)
    # The model expects trajectories in the ego frame at t0.
    # Transformation: xyz_local = R_t0^{-1} @ (xyz_world - xyz_t0)
    t0_xyz = ego_history_xyz[-1].copy()  # Position at t0
    t0_quat = ego_history_quat[-1].copy()  # Orientation at t0
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    # Transform history positions to local frame
    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)

    # Transform future positions to local frame
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

    # Transform rotations to local frame
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    # Convert to torch tensors with batch dimensions: (B=1, n_traj_group=1, T, ...)
    ego_history_xyz_tensor = (
        torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_history_rot_tensor = (
        torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    return ego_history_xyz_tensor, ego_history_rot_tensor, ego_future_xyz_tensor, ego_future_rot_tensor


def get_images_from_sample(tstamp, camera_features, config):

    image_frames_list = []

    image_timestamps = np.array(
        [tstamp - (config.num_frames - 1 - i) * int(config.time_step * 1_000_000) for i in range(config.num_frames)],
        dtype=np.int64,
    )

    for cam_feature in camera_features:
        frames, frame_timestamps = cam_feature.decode_images_from_timestamps(image_timestamps)

        # Convert to (num_frames, 3, H, W) for model input
        frames_tensor = torch.from_numpy(frames)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

        # Extend the list with individual frame tensors of shape (3, H, W)
        image_frames_list.extend(list(frames_tensor))
        
    return image_frames_list