import os
import torch
import copy

def create_separated_datasets(source_dir, base_target_dir):
    print(f"Reading from: {source_dir}")
    
    # Define the new output directories
    img_base_dir = os.path.join(base_target_dir, "image_embeddings")
    txt_base_dir = os.path.join(base_target_dir, "text_embeddings")
    
    for split in ['train', 'test', 'val']:
        source_split_path = os.path.join(source_dir, split)
        if not os.path.exists(source_split_path):
            continue
            
        # Create corresponding train/test/val folders in the new directories
        img_split_dir = os.path.join(img_base_dir, split)
        txt_split_dir = os.path.join(txt_base_dir, split)
        os.makedirs(img_split_dir, exist_ok=True)
        os.makedirs(txt_split_dir, exist_ok=True)
        
        for file in os.listdir(source_split_path):
            if file.endswith('.pt'):
                source_file_path = os.path.join(source_split_path, file)
                
                # 1. Load the untouched original data
                data = torch.load(source_file_path, weights_only=False)
                original_embeddings = data['embeddings'] # Shape: [N, 1536]
                
                # 2. Create deep copies to safely retain labels (log_prices, indices, etc.)
                img_data = copy.deepcopy(data)
                txt_data = copy.deepcopy(data)
                
                # 3. Slice and overwrite the 'embeddings' key in the new dictionaries
                # This ensures your standard PyTorch Dataset class still works perfectly!
                img_data['embeddings'] = original_embeddings[:, :768]
                txt_data['embeddings'] = original_embeddings[:, 768:]
                
                # 4. Save to the new, separate paths
                torch.save(img_data, os.path.join(img_split_dir, file))
                torch.save(txt_data, os.path.join(txt_split_dir, file))
                
                print(f"Processed {split}/{file} -> Saved to Image and Text folders.")

    print("\nExtraction complete! Your original split_embeddings folder is untouched.")
    print(f"Image dataset created at: {img_base_dir}")
    print(f"Text dataset created at:  {txt_base_dir}")

# Run the extraction
source_path = r"C:\Price-Interval-Prediction\split_embeddings"
target_path = r"C:\Price-Interval-Prediction" # It will create folders inside here

create_separated_datasets(source_path, target_path)