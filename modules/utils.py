##============================================
## Author : MEKA Moïse Christian Junior
## Email : moise.meka@students.unibe.ch
##============================================

import numpy as np
import os 
from PIL import Image
from skimage.util import random_noise

LIST_EXTENSIONS=['png', 'jpg', 'jpeg']

def read_image(filepath:str) -> np.ndarray:
    """
    Read an image from a file path and return a numpy array 

    Input:
        filepath : Path to image as str
    
    Return:
        Numpy array with shape (height, width, channels) for RGB images or (height, width) for gray scale images
    """

    if not isinstance(filepath, str):
        return None

    try:
        image = Image.open(filepath)
    except:
        return None
    
    return np.array(image)

def check_extension(filepath:str, extension:str|list) -> bool:
    """
    Check if the file given as input has the expected extension

    Input:
        filepath: Path to a file as str
        extension: One or a list of expected extensions as str or list (eg: '.jpg', ['.png','.jpeg'])
    
    Return:
        Boolean value: True if the file has the expected extension
                       False otherwise 
    """

    file_ext = os.path.splitext(filepath)

    if file_ext in extension:
        return True
    else:
        return False

def rgb2grayscale(image:np.ndarray) -> np.ndarray:
    """
    Convert a RGB image into a gray scale 

    Input:
        image: np.ndarray object with shape (height, width, channels)
    
    Return:
        np.ndarray object with shape (height, width)
    """

    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    return 0.2989 * R + 0.5870 * G + 0.1140 * B

def normalize_image(image:np.ndarray) -> np.ndarray:
    """
    Normalize pixel values between 0 and 1

    Input:
        image: np.ndarray object with shape (height, width, channels)
    
    Return:
        np.ndarray object with the same shape but pixels between 0 and 1
    """

    return np.clip(image/255, 0, 1)

def add_noise(image, prop:float = 1):
    """
    Add noise (pepper and salt) on given image with some proportion

    Input: 
        image: np.ndarray object with (height, width, channels)
        prop: float proportion of salt and pepper 
    
    Return:
        np.ndarray object with the same shape but with pepper and salte noise
    """
    return random_noise(image, mode='s&p', amount=prop)*image
