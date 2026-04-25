##============================================
## Author : MEKA Moïse Christian Junior
## Email : moise.meka@students.unibe.ch
##============================================

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
    
    if out == 'both':
        return np.sqrt(Gradient_x**2 + Gradient_y**2) , np.arctan2(Gradient_y, Gradient_x)
    elif out == 'magnitude':
        return np.sqrt(Gradient_x**2 + Gradient_y**2)
    else:
        return np.arctan2(Gradient_y, Gradient_x)

def dog_kernel(sigma:float, half=None, on:Literal['X','Y']='X')->np.ndarray:
    
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
    
    blured_image = gaussian_filter(image, sigma=sigma)

    Gx = compute_gradient(blured_image, on='X', **kwarg)
    Gy = compute_gradient(blured_image, on='Y', **kwarg)
    return magnitude_direction(Gx, Gy)

def non_maximum_suppression(magnitude:np.ndarray, angle:np.ndarray):
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
    low_thresgold = high_threshold/2

    strong = image >= high_threshold
    weak   = (image >= low_thresgold) & (image < high_threshold)
    suppressed = image < low_thresgold

    result = np.zeros((*image.shape, 3))
    result[strong] = [0, 0.9, 0]       
    result[weak]   = [0.9, 0, 0] 
    return result

def apply_canny(image:np.ndarray, sigma: float, high_threshold:float) -> np.ndarray:
    low_thresgold = high_threshold/2
    return canny(image, sigma, low_thresgold, high_threshold)

def apply_laplacian(image):
    return convolve(image, LAPLACIAN_FILTER)

def log_kernel(sigma:float, half=None, on:Literal['X','Y']='X')->np.ndarray:
    
    if half == None:
        half = int(4*sigma)
    
    axis = np.arange(-half, half+1, dtype=float)
    X,Y = np.meshgrid(axis, axis)
    G = np.exp(-(X**2 + Y**2)/(2*sigma**2))
    log = G*(-1/np.pi*sigma**4)*(1-((X**2 + Y**2)/(2*sigma**2)))
    return log - log.mean()

