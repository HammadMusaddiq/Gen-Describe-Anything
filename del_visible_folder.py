import os
import shutil

# base_dir = "/mnt/data2/Hammad/Datasets/ReID_data/prcc/rgb/train"
base_dir = "/mnt/data2/Hammad/Datasets/ReID_data/MSMT17_V2/mask_train_v2"

# Iterate through each class identity folder
for class_id in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_id)
    
    # Ensure it's a directory
    if os.path.isdir(class_path):
        visible_path = os.path.join(class_path, "visible")
        
        # Check if the 'visible' folder exists
        if os.path.isdir(visible_path):
            shutil.rmtree(visible_path)

print("✅ All 'visible' subfolders deleted from class identities folders.")