import io
import numpy as np

from PIL import Image
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


def find_nucleus(segmented_bitmap_path):
    """Extracts nucleus, cytoplasm and background from segmented image.

    Parameters
    ----------
    segmented_bitmap_path : str
        path to bitmap image corresponding to the segmentation result.

    Returns
    -------
    nucl_img : np.ndarray
        Numpy array for the nucleus image
    cyto_img : np.ndarray
        Numpy array for the cytoplasm image
    back_img : np.ndarray
        Numpy array for the background image
    """
    with open(segmented_bitmap_path, "rb") as binary_file:
        binary_data = binary_file.read()

    image = np.array(Image.open(io.BytesIO(binary_data)))
    
    # Constructing Nucleus image
    nucl_img = np.zeros(image.shape)
    ix, iy = np.where(image == 2)
    nucl_img[ix, iy] = 1
    nucl_ix, nucl_iy = np.where(nucl_img > 0)

    # Constructing Cytoplasm image
    cyto_img = np.zeros(image.shape)
    ix, iy = np.where(image == 3)
    cyto_img[ix, iy] = 1
    cyto_ix, cyto_iy = np.where(cyto_img > 0)

    # Constructing Background image
    back_img = np.ones(image.shape)
    back_img[nucl_ix, nucl_iy] = 0
    back_img[cyto_ix, cyto_iy] = 0
    back_ix, back_iy = np.where(back_img > 0)

    return nucl_img, cyto_img, back_img
