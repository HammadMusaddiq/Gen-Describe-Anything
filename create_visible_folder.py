import os
import shutil

# base_path = "/mnt/data1/Hammad/git_repos/joy-caption/data_folder/prcc/rgb/test/C"
# base_path = "/mnt/data1/Hammad/git_repos/joy-caption/data_folder/prcc/rgb/train"
base_path = "/mnt/data1/Hammad/git_repos/joy-caption/data_folder/prcc/rgb/val"

for class_id in os.listdir(base_path):
    class_path = os.path.join(base_path, class_id)
    if not os.path.isdir(class_path):
        continue  # Skip files

    visible_path = os.path.join(class_path, "visible")
    os.makedirs(visible_path, exist_ok=True)

    for filename in os.listdir(class_path):
        file_path = os.path.join(class_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.move(file_path, os.path.join(visible_path, filename))

print("✅ All images moved into their respective 'visible' subfolders.")
