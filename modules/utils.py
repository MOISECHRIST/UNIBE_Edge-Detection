##============================================
## Author : MEKA Moïse Christian Junior
## Email : moise.meka@students.unibe.ch
##============================================

"""
Utility functions for image loading, processing, and manipulation.

This module provides common utilities such as reading images, checking file extensions,
converting RGB to grayscale, normalizing pixel values, and adding noise.
"""

import numpy as np
import os 
from PIL import Image
from skimage.util import random_noise
import streamlit as st

LIST_EXTENSIONS=['png', 'jpg', 'jpeg']

sample_images = {
    "lenna":"sample_data/Lenna.png",
    "cameraman": "sample_data/cameraman.png",
    "cat1": "sample_data/cat3.png",
    "checkerboard" : "sample_data/checkerboard.png",
    "circles": "sample_data/circles.jpg",
    "cat2": "sample_data/hugo.jpg",
    "owl": "sample_data/owl.jpg",
    "shapes" : "sample_data/shapes.jpg",
    "Upload your image": None
}

def wipe_everything():
    st.session_state.clear()

def read_image(filepath:str) -> np.ndarray:
    """
    Read an image from a file path and return it as a numpy array.

    Parameters
    ----------
    filepath : str | Image Object
        Path to the image file.

    Returns
    -------
    np.ndarray or None
        Numpy array with shape (H, W, C) for RGB or (H, W) for grayscale.
        Returns None if the path is invalid or the file cannot be opened.
    """

    try:
        # The 'with' statement ensures the file is automatically closed
        with Image.open(filepath) as image:
            return np.array(image)
    except (FileNotFoundError, IOError, ValueError) as e:
        return None

def check_extension(filepath:str, extension:str|list) -> bool:
    """
    Check if the file has one of the expected extensions.

    Parameters
    ----------
    filepath : str
        Path to a file.
    extension : str or list of str
        The expected extension(s) (e.g., '.jpg' or ['.png', '.jpeg']).

    Returns
    -------
    bool
        True if the file extension matches, False otherwise.
    """

    file_ext = os.path.splitext(filepath)[1]

    if isinstance(extension, list):
        return file_ext in extension
    return file_ext == extension


def rgb2grayscale(image:np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.

    Parameters
    ----------
    image : np.ndarray
        RGB image array with shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Grayscale image array with shape (H, W).
    """
    if image.ndim==3:
        grayscale = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        grayscale = grayscale.astype(image.dtype)
        return grayscale
    return image

def normalize_image(image:np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to the range [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Input image array.

    Returns
    -------
    np.ndarray
        Normalized image array with values between 0 and 1.
    """

    return np.clip(image/255, 0, 1)

def add_noise(image: np.ndarray, prop: float = 1) -> np.ndarray:
    """
    Add salt and pepper noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image array.
    prop : float, optional
        Proportion of pixels to replace with noise (default is 1.0).

    Returns
    -------
    np.ndarray
        Image array with added salt and pepper noise.
    """
    return random_noise(image, mode='s&p', amount=prop)*image


