import os
import data
import models
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description="Augments each image on the dataset")
parser.add_argument("--test_data_path", type=str, default="./data/database/tmp/Valid")
parser.add_argument("--model_folder", type=str, default="./results/SavedModel")
args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

testGen = data.ClassificationGenerator(dataset_root_path=args.test_data_path)

model_files = os.listdir(args.model_folder)
max_acc = - np.inf
max_f = None
for f in model_files:
    acc = float(f.split('.hdf5')[0].split('weights.')[1][-4:])
    if acc > max_acc:
        max_file = f
        max_acc = acc

m = tf.keras.metrics.Accuracy()
accs = []
model = tf.keras.models.load_model(os.path.join(args.model_folder, f))
for i, (x, ytrue) in enumerate(testGen):
    ypred = model.predict(x)
    ytrue = np.argmax(ytrue, axis=1)
    ypred = np.argmax(ypred, axis=1)
    print(ytrue, ypred)
    acc = sum((ytrue) == ypred) / len(ytrue)
    accs.append(acc)
    print("Mean accuracy on batch {}: {}".format(i, acc))

print("Mean accuracy: {}".format(np.mean(accs)))
