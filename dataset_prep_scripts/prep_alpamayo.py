from typing import Any
import os
import numpy as np
import zipfile
import physical_ai_av
import scipy.spatial.transform as spt
from huggingface_hub import login
import torch
from datasets import Dataset
from torchvision import transforms
from einops import rearrange


to_pil = transforms.ToPILImage()

def load_physical_aiavdataset(
    clip_id: str,
    t0_us: int = 5_100_000,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None = None,
    maybe_stream: bool = True,
    cache_dir: str | None = None,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: list | None = None,
    num_frames: int = 4,
    skip_cameras: bool = False,
    skip_traj: bool = False,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Load data from physical_ai_av for model inference.

    This function loads a sample from the physical_ai_av dataset and converts it
    to the format expected by AlpamayoR1 model inference.

    Args:
        clip_id: The clip ID to load data from. Can be obtained from vla_golden.parquet.
        t0_us: The timestamp (in microseconds) at which to sample the trajectory.
            If None, uses a timestamp 5.1s seconds into the clip.
        avdi: Optional pre-initialized PhysicalAIAVDatasetInterface. If None, creates one.
        maybe_stream: Whether to stream data from HuggingFace (if not downloaded locally).
        num_history_steps: Number of history trajectory steps (default: 16 for 1.6s at 10Hz).
        num_future_steps: Number of future trajectory steps (default: 64 for 6.4s at 10Hz).
        time_step: Time step between trajectory points in seconds (default: 0.1s = 10Hz).
        camera_features: List of camera features to load. If None, uses 4 cameras:
            [CAMERA_FRONT_WIDE_120FOV, CAMERA_FRONT_TELE_30FOV,
             CAMERA_CROSS_LEFT_120FOV, CAMERA_CROSS_RIGHT_120FOV].
        num_frames: Number of frames per camera to load (default: 4).
        skip_cameras: Whether to skip loading camera images.
        skip_traj: Whether to skip loading trajectory data.

    Returns:
        A dictionary with the following keys (varies depending on skip flags):
            - image_frames: torch.Tensor of shape (N_cameras, num_frames, 3, H, W)
            - camera_indices: torch.Tensor of shape (N_cameras,)
            - ego_history_xyz: torch.Tensor of shape (1, 1, num_history_steps, 3)
            - ego_history_rot: torch.Tensor of shape (1, 1, num_history_steps, 3, 3)
            - ego_future_xyz: torch.Tensor of shape (1, 1, num_future_steps, 3)
            - ego_future_rot: torch.Tensor of shape (1, 1, num_future_steps, 3, 3)
            - relative_timestamps: torch.Tensor of shape (N_cameras, num_frames)
            - absolute_timestamps: torch.Tensor of shape (N_cameras, num_frames)
            - t0_us: The t0 timestamp used
            - clip_id: The clip ID
    """
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=cache_dir)
       #  avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    if not skip_cameras and camera_features is None:
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    if not skip_traj:
        # Load egomotion data
        egomotion = avdi.get_clip_feature(
            clip_id,
            avdi.features.LABELS.EGOMOTION,
            maybe_stream=maybe_stream,
        )

        assert t0_us > num_history_steps * time_step * 1_000_000, (
            "t0_us must be greater than the history time range"
        )

    # Compute timestamps for trajectory sampling
    # History: [..., t0-0.2s, t0-0.1s, t0] (num_history_steps points ending at t0)
    # Future: [t0+0.1s, t0+0.2s, ..., t0+6.4s] (num_future_steps points after t0)
    
        history_offsets_us = np.arange(
            -(num_history_steps - 1) * time_step * 1_000_000,
            time_step * 1_000_000 / 2,
            time_step * 1_000_000,
        ).astype(np.int64)
        history_timestamps = t0_us + history_offsets_us

        # Get egomotion at history and future timestamps
        ego_history = egomotion(history_timestamps)
        ego_history_xyz = ego_history.pose.translation  # (num_history_steps, 3)
        ego_history_quat = ego_history.pose.rotation.as_quat()  # (num_history_steps, 4)
    
   
        future_offsets_us = np.arange(
            time_step * 1_000_000,
            (num_future_steps + 0.5) * time_step * 1_000_000,
            time_step * 1_000_000,
        ).astype(np.int64)
        future_timestamps = t0_us + future_offsets_us

        ego_future = egomotion(future_timestamps)
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
    else:
        ego_history_xyz_tensor = None
        ego_history_rot_tensor = None
        ego_future_xyz_tensor = None
        ego_future_rot_tensor = None

    if not skip_cameras:
        # Load camera images
        image_frames_list = []
        camera_indices_list = []
        timestamps_list = []
        image_paths = {}

        # Image timestamps: if num_frames=4, load at [t0-0.3s, t0-0.2s, t0-0.1s, t0]
        image_timestamps = np.array(
            [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
            dtype=np.int64,
        )


        for cam_feature in camera_features:
            camera = avdi.get_clip_feature(
                clip_id,
                cam_feature,
                maybe_stream=maybe_stream,
            )

            # frames: (num_frames, H, W, 3) uint8
            frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)

            # Convert to (num_frames, 3, H, W) for model input
            frames_tensor = torch.from_numpy(frames)
            frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

            # Extract camera name from feature path
            if isinstance(cam_feature, str):
                cam_name = cam_feature.split("/")[-1] if "/" in cam_feature else cam_feature
                cam_name = cam_name.lower()
            else:
                raise ValueError(f"Unexpected camera feature type: {type(cam_feature)}")
            cam_idx = camera_name_to_index.get(cam_name, 0)

            image_frames_list.append(frames_tensor)
            image_paths[cam_name] = [os.path.join(clip_id, cam_name, str(ft) + ".png") for ft in frame_timestamps]
            camera_indices_list.append(cam_idx)
            timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))

        # Stack and sort by camera index for consistent ordering
        image_frames = torch.stack(image_frames_list, dim=0)  # (N_cameras, num_frames, 3, H, W)
        camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)  # (N_cameras,)
        all_timestamps = torch.stack(timestamps_list, dim=0)  # (N_cameras, num_frames)

        # Sort by camera index to ensure consistent ordering [0, 1, 2, 6] instead of arbitrary order
        sort_order = torch.argsort(camera_indices)
        image_frames = image_frames[sort_order]
        camera_indices = camera_indices[sort_order]
        all_timestamps = all_timestamps[sort_order]

        # sort image paths wrt. camera_name_to_index
        image_paths = {k: v for k, v in sorted(image_paths.items(), key=lambda item: camera_name_to_index[item[0]])}

        # Compute relative timestamps in seconds
        camera_tmin = all_timestamps.min()
        relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6  # (N_cameras, num_frames)

    else:
        image_frames = None
        camera_indices = None
        all_timestamps = None
        relative_timestamps = None
        image_paths = None

    return {
        "camera_indices": camera_indices,  # (N_cameras,)
        "ego_history_xyz": ego_history_xyz_tensor,  # (1, 1, num_history_steps, 3)
        "ego_history_rot": ego_history_rot_tensor,  # (1, 1, num_history_steps, 3, 3)
        "ego_future_xyz": ego_future_xyz_tensor,  # (1, 1, num_future_steps, 3)
        "ego_future_rot": ego_future_rot_tensor,  # (1, 1, num_future_steps, 3, 3)
        "relative_timestamps": relative_timestamps,  # (N_cameras, num_frames)
        "absolute_timestamps": all_timestamps,  # (N_cameras, num_frames)
        "image_paths": image_paths,  # (N_cameras, N_paths)
        "t0_us": t0_us,
        "clip_id": clip_id,
    }, image_frames

def save_images_locally(image_paths, image_frames, save_dir):
    cam_id = 0 
    for cam_name, paths in image_paths.items():
        for path, frame in zip(paths, image_frames[cam_id]):
            os.makedirs(os.path.join(save_dir, os.path.dirname(path)), exist_ok=True)
            to_pil(frame).save(os.path.join(save_dir, path))
        cam_id += 1

def find_usable_clip_ids(nvaiav, start_id, end_id):
    clip_ids = nvaiav.clip_index.index.values.tolist()
    chunk_ids = nvaiav.clip_index["chunk"].values.tolist()
    new_clip_ids = []
    for clip, chunk in zip(clip_ids, chunk_ids):
        if start_id <= chunk <= end_id:
            new_clip_ids.append(clip)
    return new_clip_ids

import concurrent.futures
from tqdm import tqdm

def process_clip(clip_id, bins, dataset_root, save_dir, local_avdi, skip_cameras, skip_traj):
    """
    Process a single clip: sample timepoints, load data, and save images.
    Returns a list of samples collected from this clip.
    """
    clip_samples = []
    
    # Initialize dataset interface inside the worker process
    # This avoids pickling issues and ensures thread safety for the interface
    # try:
    #     # We need a local instance of the dataset interface for each process
    #     # Assuming cache_dir is thread-safe or read-only effectively
    #     local_avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=dataset_root)
    # except Exception as e:
    #     print(f"Error initializing dataset interface for clip {clip_id}: {e}")
    #     return []

    # Generate random timestamps
    t0_us_list = np.random.uniform(bins[:-1], bins[1:])

    try:
        # print(f"Loading clip {clip_id}...")
        for curr_t0 in t0_us_list:
            # Pass the local_avdi to load_physical_aiavdataset
            sample, image_frames = load_physical_aiavdataset(
                clip_id, 
                t0_us=curr_t0, 
                avdi=local_avdi, # Use the locally created instance
                maybe_stream=False, 
                cache_dir=dataset_root,
                skip_cameras=skip_cameras,
                skip_traj=skip_traj
            )
            if not skip_cameras:
                # update image paths keys to be relative to save_dir
                for key in sample["image_paths"]:
                    sample["image_paths"][key] = [os.path.join("images", p) for p in sample["image_paths"][key]]
                save_images_locally(sample["image_paths"], image_frames, save_dir)
            clip_samples.append(sample)
        # print(f"Finished with {clip_id}")
    except Exception as e:
        print(f"Error loading clip {clip_id}: {e}")
    
    return clip_samples

if __name__ == "__main__":
    
    dataset_root = "/media/vishal/datasets/ar1_coc_dataset/"
    save_dir = "/media/vishal/datasets/ar1_coc_dataset/extracted"
    skip_cameras = False
    skip_traj = False
    login(token="")
    chunk_start = 0
    chunk_end = 0
    all_samples = []
    
    # We use a temporary interface just to find clip IDs
    nvaiav = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=dataset_root)
    
    bins = np.linspace(2_100_000, 13_100_000, 10)
    clip_ids = find_usable_clip_ids(nvaiav, chunk_start, chunk_end)
    print("found", len(clip_ids), "clips")
    
    # Number of parallel workers
    # Using threads as requested to avoid process crashes
    max_workers = 12
    
    print(f"Starting processing with {max_workers} threads...")

    try:
    # We need a local instance of the dataset interface for each process
    # Assuming cache_dir is thread-safe or read-only effectively
        local_avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=dataset_root)
    except Exception as e:
        print(f"Error initializing dataset interface for clip {clip_id}: {e}")
        
    # Use ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_clip, clip_id, bins, dataset_root, save_dir, local_avdi, skip_cameras, skip_traj): clip_id for clip_id in clip_ids}
        
        # Process results as they complete with a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(clip_ids), desc="Processing Clips"):
            clip_id = futures[future]
            try:
                result = future.result()
                all_samples.extend(result)
            except Exception as e:
                print(f"Clip {clip_id} generated an exception: {e}")

    # create dataset
    print(f"Dataset processing complete. Collected {len(all_samples)} samples.")
    if all_samples:
        ar1_hf_dataset = Dataset.from_list(all_samples)
        ar1_hf_dataset.save_to_disk(os.path.join(save_dir, "ar1_hf_dataset2"))
        print(f"Dataset saved to {os.path.join(save_dir, 'ar1_hf_dataset2')}")
    else:
        print("No samples collected. Dataset not saved.")