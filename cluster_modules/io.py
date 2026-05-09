import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

def load_png(file_path: str, scale: float = 1.0, alpha: float = 1.0):
    """Load a color PNG image and return it as a numpy array, with scaling and alpha."""
    img_array = mpimg.imread(file_path)  # shape: (H, W, 3) or (H, W, 4)
    if img_array.ndim == 2:  # Grayscale image
        img_array = np.stack((img_array,) * 3, axis=-1)  # Convert to RGB
    # Only scale height and width, not channels
    if scale != 1.0:
        img_array = zoom(img_array, (scale, scale, 1), order=1)
    # If overlay is not RGBA, convert to RGBA
    if img_array.shape[-1] == 3:
        alpha_channel = np.ones(img_array.shape[:2], dtype=img_array.dtype) * alpha
        img_array = np.dstack((img_array, alpha_channel))
    else:
        img_array[..., 3] = alpha  # Set alpha for all pixels
    return img_array


def show_2_pngs(image1, image2, title1=None, title2=None):
    """Display two color images side by side using matplotlib."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    if image1.shape[-1] == 4:  # RGBA
        rgb1 = image1[..., :3]
        alpha1 = image1[..., 3:4]
        white1 = np.ones_like(rgb1)
        img1 = alpha1 * rgb1 + (1 - alpha1) * white1
    else:
        img1 = image1
    
    if image2.shape[-1] == 4:  # RGBA
        rgb2 = image2[..., :3]
        alpha2 = image2[..., 3:4]
        white2 = np.ones_like(rgb2)
        img2 = alpha2 * rgb2 + (1 - alpha2) * white2
    else:
        img2 = image2
    
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_png(image_array, title=None):
    """Display a color image using matplotlib, compositing transparent pixels over white."""
    if image_array.shape[-1] == 4:  # RGBA
        rgb = image_array[..., :3]
        alpha = image_array[..., 3:4]
        white = np.ones_like(rgb)
        # Composite over white: out = alpha*rgb + (1-alpha)*white
        img = alpha * rgb + (1 - alpha) * white
    else:
        img = image_array
    plt.imshow(img, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
