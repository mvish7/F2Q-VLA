import os
import argparse
import numpy as np
import concurrent.futures
import physical_ai_av
from tqdm import tqdm
from huggingface_hub import login
from datasets import Dataset

def find_usable_clip_ids(nvaiav, start_id, end_id):
    clip_ids = nvaiav.clip_index.index.values.tolist()
    chunk_ids = nvaiav.clip_index["chunk"].values.tolist()
    new_clip_ids = []
    for clip, chunk in zip(clip_ids, chunk_ids):
        if start_id <= chunk <= end_id:
            new_clip_ids.append(clip)
    return new_clip_ids

def process_clip(clip_id, bins):
    """
    Process a single clip: sample timepoints, load data, and save images.
    Returns a list of samples collected from this clip.
    """
    # Generate random timestamps
    t0_us_list = np.random.uniform(bins[:-1], bins[1:])
    
    all_samples = []
    for t_curr in t0_us_list:
        sample = {
            "clip_id": clip_id,
            "t_curr": t_curr,
        }
        all_samples.append(sample)
    return all_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract samples from physical_ai_av dataset.")
    parser.add_argument("--dataset_root", type=str, default="/media/vishal/datasets/ar1_coc_dataset/", help="Path to the physical_ai_av dataset cache")
    parser.add_argument("--save_dir", type=str, default="/media/vishal/datasets/ar1_coc_dataset/extracted", help="Directory to save the extracted dataset")
    parser.add_argument("--hf_token", type=str, default="", help="Hugging Face token for authentication")
    parser.add_argument("--chunk_start", type=int, default=0, help="Start chunk for finding clips")
    parser.add_argument("--chunk_end", type=int, default=5, help="End chunk for finding clips")
    parser.add_argument("--max_workers", type=int, default=12, help="Number of max workers for parallel execution")
    parser.add_argument("--num_samples_per_clip", type=int, default=10, help="Number of samples to extract per clip")
    args = parser.parse_args()
    
    dataset_root = args.dataset_root
    save_dir = args.save_dir
    login(token=args.hf_token)
    chunk_start = args.chunk_start
    chunk_end = args.chunk_end
    max_workers = args.max_workers
    all_samples = []
    
    # We use a temporary interface just to find clip IDs
    nvaiav = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=dataset_root)
    
    bins = np.linspace(2_100_000, 13_100_000, args.num_samples_per_clip)
    clip_ids = find_usable_clip_ids(nvaiav, chunk_start, chunk_end)
    print("found", len(clip_ids), "clips")
    
    # Number of parallel workers
    # Using threads as requested to avoid process crashes
    print(f"Starting processing with {max_workers} threads...")

    # Use ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_clip, clip_id, bins): clip_id for clip_id in clip_ids}
        
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
        os.makedirs(save_dir, exist_ok=True)
        ar1_hf_dataset.save_to_disk(os.path.join(save_dir, "ar1_hf_dataset_new"))
        print(f"Dataset saved to {os.path.join(save_dir, 'ar1_hf_dataset_new')}")
    else:
        print("No samples collected. Dataset not saved.")