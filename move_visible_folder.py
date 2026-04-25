import os
import shutil

def move_visible_folders(source_root, target_root):
    # Loop through each class ID folder in source
    for class_id in os.listdir(source_root):
        source_class_path = os.path.join(source_root, class_id)
        target_class_path = os.path.join(target_root, class_id)

        visible_folder = os.path.join(source_class_path, "visible")

        # Skip if no visible folder exists
        if not os.path.isdir(visible_folder):
            print(f"⚠️ No 'visible' folder found in {source_class_path}")
            continue

        # Ensure the destination class folder exists
        if not os.path.exists(target_class_path):
            print(f"❌ Target class folder does not exist: {target_class_path}")
            continue

        # Move the visible folder
        # print(f"Moving '{visible_folder}' → '{target_visible_folder}'")
        shutil.move(visible_folder, target_class_path)

if __name__ == "__main__":
    source_root = "/mnt/data1/Hammad/git_repos/joy-caption/data_folder/MSMT17_V2-/mask_train_v2"
    target_root = "/mnt/data1/Hammad/git_repos/joy-caption/data_folder/MSMT17_V2/mask_train_v2"

    move_visible_folders(source_root, target_root)
