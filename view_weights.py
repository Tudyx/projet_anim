import os
import data
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_conv_actv(image, filename, figsize=(10,5)):
    _, _, _, n_channels = image.shape
    fig, axis = plt.subplots(8, n_channels // 8)
    print(len(axis.flatten()))
    _image = np.squeeze(image)

    for chan, ax in enumerate(axis.flatten()):
        ax.imshow(_image[:, :, chan], cmap='gray')
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)


def plot_conv_filters(filters, filename, figsize=(10, 5)):
    hfilt, wfilt, cin, cout = filters.shape
    _filters = filters.reshape([hfilt, wfilt, -1])
    fig, axis = plt.subplots(cin, cout, figsize=figsize)

    for chan, ax in enumerate(axis.flatten()):
        ax.imshow(_filters[:, :, chan], cmap='gray')
        ax.set_axis_off()

    #plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)


parser = argparse.ArgumentParser(description="Augments each image on the dataset")
parser.add_argument("--test_data_path", type=str, default="./data/database/tmp/Valid")
parser.add_argument("--model_folder", type=str, default="./results/SavedModel")
args = parser.parse_args()

testGen = data.ClassificationGenerator(dataset_root_path=args.test_data_path)

model_folder = "./results/SavedModel"
model_files = os.listdir(model_folder)
max_acc = - np.inf
max_f = None
for f in model_files:
    acc = float(f.split('.hdf5')[0].split('weights.')[1][-4:])
    if acc > max_acc:
        max_file = f
        max_acc = acc

model = tf.keras.models.load_model(os.path.join(model_folder, f))

layer_actv = [layer.output for layer in model.layers[1:]]
inner_actv_model = tf.keras.models.Model(inputs=model.input, outputs=layer_actv)
