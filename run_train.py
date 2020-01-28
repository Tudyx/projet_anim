import data
import models
import argparse
import tensorflow as tf

from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

parser = argparse.ArgumentParser(description="Augments each image on the dataset")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Number of images per training batch")
parser.add_argument("--n_epochs", type=int, default=300,
                    help="Number of training epochs")
parser.add_argument("--use_batchnorm", type=bool, default=False,
                    help="Whether to use batchnorm in convolutional layer or not")
parser.add_argument("--dropout_rate", type=float, default=0.0,
                    help="Rate of units dropped at each fully connected layer")
args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

trainGen = data.ClassificationGenerator(dataset_root_path="./data/database/tmp/Train",
                                        batch_size=args.batch_size)
testGen = data.ClassificationGenerator(dataset_root_path="./data/database/tmp/Test")

model = models.small_convnet(dropout_rate=args.dropout_rate,
                             normalize=args.use_batchnorm)

adamopt = optimizers.Adam()

model.compile(optimizer=adamopt,
              loss=losses.binary_crossentropy,
              metrics=[
                  "accuracy",# metrics.Accuracy,
                  metrics.FalsePositives(),
                  metrics.FalseNegatives(),
                  metrics.TruePositives(),
                  metrics.TrueNegatives()
              ])

model.fit_generator(generator=trainGen,
                    steps_per_epoch=180,
                    epochs=args.n_epochs,
                    callbacks=[
                        callbacks.ModelCheckpoint(filepath="./results/SavedModel/"\
                                                           "weights.{epoch:02d}-{val_acc:.2f}.hdf5",
                                                  monitor="acc",
                                                  save_best_only=True),
                        callbacks.TensorBoard(log_dir="./results/TensorBoard",
                                              histogram_freq=1,
                                              batch_size=32,
                                              write_graph=True,
                                              write_grads=True,
                                              write_images=True,
                                              update_freq="epoch")
                    ],
                    validation_data=testGen,
                    validation_freq=1)