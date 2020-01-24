import os
import shutil

dataset_root_path = os.path.join(os.path.expanduser("~"), "data/ProjAnim/martin2003/Matlab feature extraction")
folders = [
    "carcinoma_in_situ",
    "light_dysplastic",
    "moderate_dysplastic",
    "normal_columnar",
    "normal_intermediate",
    "normal_superficiel",
    "severe_dysplastic"
]

try:
    os.mkdir("../database/segmentation")
except FileExistsError as err:
    print("Warning: folder {} already exists.".format(os.path.abspath("../segmentation")))

try:
    os.mkdir("../database/segmentation/image")
except FileExistsError as err:
    print("Warning: folder {} already exists.".format(os.path.abspath("../segmentation")))

try:
    os.mkdir("../database/segmentation/label")
except FileExistsError as err:
    print("Warning: folder {} already exists.".format(os.path.abspath("../segmentation")))

for folder in folders:
    files = os.listdir(os.path.join(dataset_root_path, folder))

    for file in files:
        src_path = os.path.join(dataset_root_path, folder, file)
        if "-d" in file:
            dest_path = "../database/segmentation/label"
            print("Copying segmented image {} to {}".format(src_path, dest_path))
            shutil.copy2(src_path, dest_path)
        else:
            dest_path = "../database/segmentation/image"
            print("Copying original image {} to {}".format(src_path, dest_path))
            shutil.copy(src_path, dest_path)
