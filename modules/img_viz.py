##============================================
## Author : MEKA Moïse Christian Junior
## Email : moise.meka@students.unibe.ch
##============================================

import numpy as np
from matplotlib import pyplot as plt


def plot_gradients_img(image:np.ndarray, gradient1_img:np.ndarray, gradient2_img:np.ndarray, 
                       titles = ["Original Image", "Gx (horizontal)", "Gy (vertical)"],
                       figsize=(12, 8), cmaps=["grey","RdBu_r","RdBu_r"]):
    """
    Plot the original image and its gradients along two axes.

    Parameters
    ----------
    image : np.ndarray
        The original input image.
    gradient1_img : np.ndarray
        The gradient image along the first axis (e.g., horizontal).
    gradient2_img : np.ndarray
        The gradient image along the second axis (e.g., vertical).
    titles : list of str, optional
        Titles for the three subplots (default is ["Original Image", "Gx (horizontal)", "Gy (vertical)"]).
    figsize : tuple, optional
        The size of the figure (default is (12, 8)).
    cmaps : list of str, optional
        Colormaps for the three subplots (default is ["grey", "RdBu_r", "RdBu_r"]).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plots.
    """
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=figsize)
    axes[0].imshow(image, cmap=cmaps[0])
    axes[0].set_title(titles[0],fontsize=18)
    axes[1].imshow(gradient1_img, cmap=cmaps[1])
    axes[1].set_title(titles[1],fontsize=18)
    axes[2].imshow(gradient2_img, cmap=cmaps[2])
    axes[2].set_title(titles[2],fontsize=18)
    for ax in axes: 
        ax.axis('off')
    return fig


def plot_step_img_process(image:np.ndarray, result_img_processing:np.ndarray,
                          titles = ["Original Image", "Processing result"],
                       figsize=(12, 8), cmaps=["grey","RdBu_r"]):
    """
    Plot the original image and the result of an image processing step side-by-side.

    Parameters
    ----------
    image : np.ndarray
        The original input image.
    result_img_processing : np.ndarray
        The image after processing.
    titles : list of str, optional
        Titles for the two subplots (default is ["Original Image", "Processing result"]).
    figsize : tuple, optional
        The size of the figure (default is (12, 8)).
    cmaps : list of str, optional
        Colormaps for the two subplots (default is ["grey", "RdBu_r"]).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plots.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize)
    axes[0].imshow(image, cmap=cmaps[0])
    axes[0].set_title(titles[0],fontsize=18)
    axes[1].imshow(result_img_processing, cmap=cmaps[1])
    axes[1].set_title(titles[1],fontsize=18)
    
    for ax in axes: 
        ax.axis('off')
    
    return fig