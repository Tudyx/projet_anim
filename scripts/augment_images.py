import os
import sys
import argparse
import numpy as np
import pandas as pd
import imgaug.augmenters as iaa

sys.path.append("../")

from tqdm import tqdm
from data import crop_center
from skimage.io import imread, imsave

np.random.seed(0)

parser = argparse.ArgumentParser(description="Augments each image on the dataset")
parser.add_argument("--src_dir", type=str)
parser.add_argument("--dest_dir", type=str)
args = parser.parse_args()

dataset_root_path = args.src_dir
augmentation_output_path = args.dest_dir

print("Augmenting images from {}. Saving on {}".format(args.src_dir, args.dest_dir))

try:
    os.makedirs(augmentation_output_path)
except FileExistsError:
    if not os.listdir(augmentation_output_path):
        print("Error: folder {} already exists and is not empty. Exiting.".format(os.path.abspath(augmentation_output_path)))

label_to_name_map = {
    1: "[NORMAL] Normal Superficial",
    2: "[NORMAL] Normal Intermediate",
    3: "[NORMAL] Normal Columnar",
    4: "[ABNORM] Light Dysplastic",
    5: "[ABNORM] Moderate Dysplastic",
    6: "[ABNORM] Severe Dysplastic",
    7: "[ABNORM] Carcinoma In Situ"
}

sys.path.append(dataset_root_path)
df = pd.read_csv(os.path.join(dataset_root_path, "results.csv"))
summary = df.Class.value_counts()

nucleus_positions = {
    filename: (int(cx), int(cy))
    for filename, cx, cy in zip(df.values[:, 1], df.values[:, 2], df.values[:, 3])
}

n_abnorm = 0
n_normal = 0

print("Original dataset Summary:")
for key in summary.keys():
    if "[ABNORM]" in label_to_name_map[key]:
        n_abnorm += summary[key]
    else:
        n_normal += summary[key]
    print("{}\n... Number of samples: {}".format(label_to_name_map[key], summary[key]))
    print("--------------------------------")
print("Abnormal\n... Number of Samples: {}".format(n_abnorm))
print("Normal\n... Number of Samples: {}".format(n_normal))
print("--------------------------------\n\n")

n_abnorm = 0
n_normal = 0

print("Dataset after Data Augmentation summary:")
for key in summary.keys():
    if "[ABNORM]" in label_to_name_map[key]:
        n_aug = 100
        n_abnorm += n_aug * summary[key]
    else:
        n_aug = 280
        n_normal += n_aug * summary[key]
    print("{}\n... Number of samples: {}\nAugmentation Factor: 1:{}".format(label_to_name_map[key], n_aug * summary[key], n_aug))
    print("--------------------------------")
print("Abnormal\n... Number of Samples: {}".format(n_abnorm))
print("Normal\n... Number of Samples: {}".format(n_normal))

answer = input("\n\n--> Proceed to augment images on {}? [y/n] ".format(os.path.abspath(dataset_root_path)))

if answer.lower() == "y":
    pass
elif answer.lower() == "n":
    exit()
else:
    print("Expected to receive 'y' or 'n', but got '{}'. Quitting the script".format(answer))
    exit()

filenames = df.values[:, 1]
classes = df.values[:, -1]

augmentations = {
    1: {
        "Translation": (14, iaa.Affine(translate_px={"x": (-5, 5), "y": (-5, 5)})),
        "Rotation": (20, iaa.Affine(rotate=(-18, 18)))
    },
    2: {
        "Translation": (14, iaa.Affine(translate_px={"x": (-5, 5), "y": (-5, 5)})),
        "Rotation": (20, iaa.Affine(rotate=(-18, 18)))
    },
    3: {
        "Translation": (14, iaa.Affine(translate_px={"x": (-5, 5), "y": (-5, 5)})),
        "Rotation": (20, iaa.Affine(rotate=(-18, 18)))
    },
    4: {
        "Translation": (10, iaa.Affine(translate_px={"x": (-5, 5), "y": (-5, 5)})),
        "Rotation": (10, iaa.Affine(rotate=(-36, 36)))
    },
    5: {
        "Translation": (10, iaa.Affine(translate_px={"x": (-5, 5), "y": (-5, 5)})),
        "Rotation": (10, iaa.Affine(rotate=(-36, 36)))
    },
    6: {
        "Translation": (10, iaa.Affine(translate_px={"x": (-5, 5), "y": (-5, 5)})),
        "Rotation": (10, iaa.Affine(rotate=(-36, 36)))
    },
    7: {
        "Translation": (10, iaa.Affine(translate_px={"x": (-5, 5), "y": (-5, 5)})),
        "Rotation": (10, iaa.Affine(rotate=(-36, 36)))
    }
}

_augmented_filenames = []
_augmented_classes = []
for filename, clss in tqdm(zip(filenames, classes)):
    cx, cy = nucleus_positions[filename]
    image = imread(os.path.join(dataset_root_path, filename))
    _image = crop_center(image, cx, cy, window_size=128)

    imsave(os.path.join(augmentation_output_path, filename), _image)
    _augmented_filenames.append(filename)
    _augmented_classes.append(clss)

    augmentation_dict = augmentations[clss]
    nT, translation = augmentation_dict["Translation"]
    nR, rotation = augmentation_dict["Rotation"]

    for i in range(nT):
        for j in range(nR):
            k = i * nR + j
            augmented_filename = str(k) + "_" + filename
            _augmented_filenames.append(augmented_filename)
            _augmented_classes.append(clss)

            _image_t = translation.augment_image(_image)
            _image_r = rotation.augment_image(_image_t)

            imsave(os.path.join(augmentation_output_path, augmented_filename), _image_r)

with open("../data/database/tmp/results.csv", "w") as file:
    file.write("Image Filepath,Class\n")
    for filename, clss in zip(_augmented_filenames, _augmented_classes):
        if clss in [1, 2, 3]:
            _clss = 0
        elif clss in [4, 5, 6, 7]:
            _clss = 1
        file.write("{},{}\n".format(filename, _clss))