import argparse
from skimage.color import rgb2gray, yuv2rgb
import numpy as np
from skimage.io import imsave
from skimage.io import imread
from scipy import ndimage
import os
import shutil
from skimage import img_as_float
import matplotlib.pyplot as plt

src_path = '../data/database/classification/'
src_img_name = '148495885-148495896-001.BMP'
dest_path = '../data/database/segmented/'
dest_img_name = 'segmented_image.BMP'

# Read original image to be segmented
image = imread(os.path.join(src_path, src_img_name))

# Show original image
plt.figure()
plt.imshow(image)

# Create grayscale image
gray = rgb2gray(image)

# Show grayscale image
plt.figure()
plt.imshow(gray)
# imsave(os.path.join(dest_path, 'gray_image.BMP'), gray)

# 1-D array conversion
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])

# Computation of threshold segmentation
for i in range(gray_r.shape[0]):
    if gray_r[i] > (3/2)*gray_r.mean():
        gray_r[i] = 3
    elif gray_r[i] > gray_r.mean():
        gray_r[i] = 2
    elif gray_r[i] > (1/2)*gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0
# 2-D array conversion
gray = gray_r.reshape(gray.shape[0], gray.shape[1])

# Show segmented image
plt.figure()
plt.imshow(gray)
plt.show()

# Save segmented image
imsave(os.path.join(dest_path, dest_img_name), gray)
