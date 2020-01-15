import numpy as np

from skimage.io import imread
from skimage import img_as_float


def crop_center(image, cx, cy, window_size=120):
	"""Crop a window_size x window_size patch centered on coordinates of cellular center.

	Parameters
	----------
	image : str
		Name of file to load.
	cx : int
		Center position on x axis.
	cy : int
		Center position on y axis.
	window_size : int
		Integer defining the patch window size.
	"""
	assert image.ndim == 3, "Expecting 3D array (H, W, C) but got" \
	                        " {}D array with shape {}".format(image.ndim, image.shape)

	h, w, _ = image.shape
	
	pad_left = window_size // 2 - cx if cx - window_size // 2 < 0 else 0
	pad_up = window_size // 2 - cy if cy - window_size // 2 < 0 else 0
	pad_right = cx + window_size // 2 - h if cx + window_size // 2 > h else 0
	pad_down = cy + window_size // 2 - w if cy + window_size // 2 > w else 0
	pad = np.max([pad_left, pad_up, pad_right, pad_down])

	img_pad = np.pad(image, ((pad, pad), (pad, pad), (0, 0))) if pad != 0 else image
	cx += pad
	cy += pad 

	return img_pad[cx - window_size // 2: cx + window_size // 2, cy - window_size // 2: cy + window_size // 2, :]