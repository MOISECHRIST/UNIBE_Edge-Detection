import numpy as np
from matplotlib import pyplot as plt


def plot_gradients_img(image:np.ndarray, gradient1_img:np.ndarray, gradient2_img:np.ndarray, 
                       titles = ["Original Image", "Gx (horizontal)", "Gy (vertical)"],
                       figsize=(12, 8), cmaps=["grey","RdBu_r","RdBu_r"]):
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=figsize)
    axes[0].imshow(image, cmap=cmaps[0])
    axes[0].set_title(titles[0])
    axes[1].imshow(gradient1_img, cmap=cmaps[1])
    axes[1].set_title(titles[1])
    axes[2].imshow(gradient2_img, cmap=cmaps[2])
    axes[2].set_title(titles[2])
    for ax in axes: 
        ax.axis('off')
    return fig