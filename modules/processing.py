##============================================
## Author : MEKA Moïse Christian Junior
## Email : moise.meka@students.unibe.ch
##============================================

"""
Core image processing functions for edge detection and filtering.

This module implements various edge detection algorithms and filters including:
- Gradient computation (Sobel, Prewitt, Roberts)
- Difference of Gaussians (DoG)
- Laplacian of Gaussian (LoG)
- Canny edge detection steps (NMS, Hysteresis thresholding)
"""

import numpy as np 
from scipy.ndimage import convolve, gaussian_filter
from skimage.feature import canny
from typing import Literal

FILTERS = {
    'sobel_X': np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
]),
    'prewitt_X': np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
]),
    'robert_X': np.array([
    [1,0],
    [0,-1]
]),
    'robert_Y': np.array([
    [0,1],
    [-1,0]
])}

LAPLACIAN_FILTER = np.array([
    [0,1,0],
    [1,-4,1],
    [0,1,0]
])

def compute_gradient(image:np.ndarray, on:Literal['X', 'Y'] = 'X', filtername:Literal['sobel', 'prewitt', 'robert'] = 'sobel')-> np.ndarray:
    """
    Compute the image gradient along a specified axis using a chosen filter.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.
    on : {'X', 'Y'}, optional
        The axis along which to compute the gradient (default is 'X').
    filtername : {'sobel', 'prewitt', 'robert'}, optional
        The type of filter to use for gradient computation (default is 'sobel').

    Returns
    -------
    np.ndarray or None
        The computed gradient image, or None if parameters are invalid.
    """
    
    if filtername in ['sobel', 'prewitt']:
        if on == 'X':
            return convolve(image, FILTERS[f"{filtername}_{on}"])
        elif on == 'Y':
            return convolve(image, FILTERS[f"{filtername}_X"].T)
        else:
            return None
    elif filtername == 'robert':
        return convolve(image, FILTERS[f"{filtername}_{on}"])
    else:
        return None

def magnitude_direction(Gradient_x: np.ndarray, Gradient_y: np.ndarray, out: Literal['both', 'magnitude', 'orientation']='both') -> tuple|np.ndarray:
    """
    Calculate the magnitude and/or orientation of the image gradient.

    Parameters
    ----------
    Gradient_x : np.ndarray
        Gradient of the image along the X-axis.
    Gradient_y : np.ndarray
        Gradient of the image along the Y-axis.
    out : {'both', 'magnitude', 'orientation'}, optional
        Specifies which values to return (default is 'both').

    Returns
    -------
    tuple or np.ndarray
        - If 'both': (magnitude, orientation)
        - If 'magnitude': magnitude array
        - If 'orientation': orientation (angle) array in radians.
    """
    
    if out == 'both':
        return np.sqrt(Gradient_x**2 + Gradient_y**2) , np.arctan2(Gradient_y, Gradient_x)
    elif out == 'magnitude':
        return np.sqrt(Gradient_x**2 + Gradient_y**2)
    else:
        return np.arctan2(Gradient_y, Gradient_x)

def dog_kernel(sigma:float, half=None, on:Literal['X','Y']='X')->np.ndarray:
    """
    Generate a Derivative of Gaussians (DoG) kernel (derivative of Gaussian).

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian distribution.
    half : int, optional
        Half-width of the kernel. If None, it is calculated as 3*sigma.
    on : {'X', 'Y'}, optional
        The axis along which to compute the derivative (default is 'X').

    Returns
    -------
    np.ndarray
        The generated DoG kernel.
    """
    
    if half == None:
        half = int(3*sigma)
    
    axis = np.arange(-half, half+1, dtype=float)
    X,Y = np.meshgrid(axis, axis)
    G = np.exp(-(X**2 + Y**2)/(2*sigma**2))
    
    if on == 'X':
        dgdx = G*(-X/(2*np.pi*sigma**4))
        return dgdx - dgdx.mean()
    
    else:
        dgdy = G*(-Y/(2*np.pi*sigma**4))
        return dgdy - dgdy.mean()

def gradient_of_gaussian(image:np.ndarray, sigma:float, **kwarg) -> tuple:
    """
    Compute the gradient of an image after applying Gaussian smoothing.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.
    sigma : float
        Standard deviation for the Gaussian filter.
    **kwarg : dict
        Additional arguments passed to `compute_gradient`.

    Returns
    -------
    tuple
        (magnitude, orientation) of the gradient.
    """
    
    blured_image = gaussian_filter(image, sigma=sigma)

    Gx = compute_gradient(blured_image, on='X', **kwarg)
    Gy = compute_gradient(blured_image, on='Y', **kwarg)
    mag, direct =  magnitude_direction(Gx, Gy)
    return Gx, Gy, mag, direct

def non_maximum_suppression(magnitude:np.ndarray, angle:np.ndarray):
    """
    Perform Non-Maximum Suppression (NMS) to thin edges.

    Parameters
    ----------
    magnitude : np.ndarray
        Gradient magnitude of the image.
    angle : np.ndarray
        Gradient orientation in radians.

    Returns
    -------
    np.ndarray
        Image with thinned edges.
    """
    height, width = magnitude.shape
    result = np.zeros_like(magnitude)
    angle_deg = np.rad2deg(angle) % 180
    for i in range(1, height-1):
        for j in range(1, width-1):
            if (0 <= angle_deg[i,j] < 22.5) or (157.5 <=  angle_deg[i,j] < 180):
                neighbours= [magnitude[i, j-1], magnitude[i, j+1]]
            elif 22.5 <= angle_deg[i,j] < 67.5:
                neighbours= [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            elif 67.5 <= angle_deg[i,j] < 112.5:
                neighbours= [magnitude[i-1, j], magnitude[i+1, j]]
            else:
                neighbours= [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            result[i,j] = (np.max(neighbours)<= magnitude[i,j])*magnitude[i,j]
    return result

def show_hyteresis_thresholding(image:np.ndarray, high_threshold:float):
    """
    Visualize strong and weak edges based on hysteresis thresholding.

    Strong edges are pixels above the high threshold.
    Weak edges are pixels between low (high/2) and high thresholds.

    Parameters
    ----------
    image : np.ndarray
        Input thinned edge image (from NMS).
    high_threshold : float
        The high threshold for hysteresis.

    Returns
    -------
    np.ndarray
        An RGB image visualizing strong edges (green) and weak edges (red).
    """
    low_thresgold = high_threshold/2

    strong = image >= high_threshold
    weak   = (image >= low_thresgold) & (image < high_threshold)

    result = np.zeros((*image.shape, 3))
    result[strong] = [0, 0.9, 0]       
    result[weak]   = [0.9, 0, 0] 
    return result

def apply_canny(image:np.ndarray, sigma: float, high_threshold:float) -> np.ndarray:
    """
    Apply Canny edge detection using the scikit-image implementation.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.
    sigma : float
        Standard deviation of the Gaussian filter.
    high_threshold : float
        The high threshold for hysteresis.

    Returns
    -------
    np.ndarray
        Boolean array where True indicates an edge.
    """
    low_thresgold = high_threshold/2
    return canny(image, sigma, low_thresgold, high_threshold)

def apply_laplacian(image: np.ndarray) -> np.ndarray:
    """
    Apply a Laplacian filter for edge detection (second-order derivative).

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.

    Returns
    -------
    np.ndarray
        The filtered image showing second-order derivatives.
    """
    return convolve(image, LAPLACIAN_FILTER)

def log_kernel(sigma:float, half=None, on:Literal['X','Y']='X')->np.ndarray:
    """
    Generate a Laplacian of Gaussian (LoG) kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian distribution.
    half : int, optional
        Half-width of the kernel. If None, it is calculated as 4*sigma.
    on : {'X', 'Y'}, optional
        Unused parameter kept for consistency with DoG signature.

    Returns
    -------
    np.ndarray
        The generated LoG kernel.
    """
    
    if half == None:
        half = int(4*sigma)
    
    axis = np.arange(-half, half+1, dtype=float)
    X,Y = np.meshgrid(axis, axis)
    G = np.exp(-(X**2 + Y**2)/(2*sigma**2))
    log = G*(-1/np.pi*sigma**4)*(1-((X**2 + Y**2)/(2*sigma**2)))
    return log - log.mean()

