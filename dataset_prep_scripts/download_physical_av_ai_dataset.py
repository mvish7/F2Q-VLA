import argparse
import os
from huggingface_hub import HfApi, hf_hub_download
from collections import defaultdict
import fnmatch

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"

TARGET_CAMERA_FOLDERS = [
    "camera/camera_front_wide_120fov",
    "camera/camera_front_tele_30fov",
    "camera/camera_cross_left_120fov",
    "camera/camera_cross_right_120fov"
]

# TARGET_CAMERA_FOLDERS = []

LABELS_FOLDER = "labels/egomotion"

def get_parser():
    parser = argparse.ArgumentParser(description="Download subset of NVIDIA PhysicalAI dataset")
    parser.add_argument("--start_chunk_id", type=int, default=60, help="Start chunk ID (inclusive)")
    parser.add_argument("--end_chunk_id", type=int, default=120, help="End chunk ID (inclusive)")
    parser.add_argument("--output_dir", "-o", type=str, default="data/nvidia_physical_av", help="Local directory to download to")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be downloaded without downloading")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (optional, picks up from env or login)")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    api = HfApi(token=args.token)
    
    print(f"Listing files in {REPO_ID}...")
    # List all files in the repo (this might take a moment if the repo is huge, but it's the reliable way to filter)
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    
    files_to_download = []
    
    # Process Camera Files
    selected_chunk_ids = set()
    for folder in TARGET_CAMERA_FOLDERS:
        # Filter files belonging to this folder and ending in .zip
        # We look for exact prefix match to ensure we represent the structure correctly
        folder_files = [f for f in all_files if f.startswith(folder + "/") and f.endswith(".zip")]
        folder_files.sort() # Ensure deterministic order
        
        selected_files = []
        for f in folder_files:
            parts = f.split(".")
            for part in parts:
                if part.startswith("chunk_") and part[6:].isdigit():
                    cid = int(part[6:])
                    if args.start_chunk_id <= cid <= args.end_chunk_id:
                        selected_files.append(f)
                        selected_chunk_ids.add(part)
                    break

        print(f"Found {len(folder_files)} files in {folder}. Selecting {len(selected_files)} (Chunks {args.start_chunk_id}-{args.end_chunk_id}).")
        for f in selected_files:
            if not os.path.exists(os.path.join(args.output_dir, f)):
                files_to_download.append(f)
    
    print(f"Selected {len(selected_chunk_ids)} unique chunks: {sorted(list(selected_chunk_ids))}")

    # Process Labels
    # Format: labels/egomotion/egomotion.chunk_XXXX.zip

    # files_to_download = []
    label_files = [
        f for f in all_files 
        if f.startswith(LABELS_FOLDER + "/") and any(chunk_id in f for chunk_id in selected_chunk_ids)
    ]
    print(f"Found {len(label_files)} matching label files in {LABELS_FOLDER}.")
    for f in label_files:
        if not os.path.exists(os.path.join(args.output_dir, f)):
            files_to_download.append(f)
    
    print(f"\nTotal files to download: {len(files_to_download)}")
    
    if args.dry_run:
        print("\n[Dry Run] Files that would be downloaded:")
        for f in files_to_download:
            print(f" - {f}")
        return

    # Download
    print(f"\nDownloading to {args.output_dir}...")
    for file_path in files_to_download:
        print(f"Downloading {file_path}...")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                repo_type="dataset",
                local_dir=args.output_dir,
                token=args.token
            )
        except Exception as e:
            print(f"Error downloading {file_path}: {e}")

    print("\nDownload complete!")

if __name__ == "__main__":
    main()
