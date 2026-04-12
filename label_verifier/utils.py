import cv2
import numpy as np
from PIL import Image
import os


def pil_to_cv(img):
    """
    Convert a PIL Image to an OpenCV BGR numpy array.
    Args:
        img (PIL.Image): Input image.
    Returns:
        np.ndarray: BGR image.
    """
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(img):
    """
    Convert an OpenCV BGR numpy array to a PIL Image.
    Args:
        img (np.ndarray): BGR image.
    Returns:
        PIL.Image: PIL image.
    """
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def bgr_to_rgb(img):
    """
    Convert an OpenCV BGR numpy array to RGB.
    Args:
        img (np.ndarray): BGR image.
    Returns:
        np.ndarray: RGB image.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img):
    """
    Convert an RGB numpy array to OpenCV BGR.
    Args:
        img (np.ndarray): RGB image.
    Returns:
        np.ndarray: BGR image.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr_to_pil(img):
    """
    Convert an OpenCV BGR numpy array to a PIL Image (RGB).
    Args:
        img (np.ndarray): BGR image.
    Returns:
        PIL.Image: PIL image in RGB mode.
    """
    return Image.fromarray(bgr_to_rgb(img))


def pil_to_bgr(img):
    """
    Convert a PIL Image to an OpenCV BGR numpy array.
    Args:
        img (PIL.Image): PIL image.
    Returns:
        np.ndarray: BGR image.
    """
    return rgb_to_bgr(np.array(img.convert('RGB')))


def crop_image(img, bbox):
    """
    Crop an image to the given bounding box.
    Args:
        img (np.ndarray): Input image.
        bbox (tuple): (x, y, w, h) bounding box.
    Returns:
        np.ndarray: Cropped image.
    """
    x, y, w, h = bbox
    return img[y:y+h, x:x+w]


def safe_makedirs(path):
    """
    Create directories if they do not exist (safe).
    Args:
        path (str): Directory path.
    """
    os.makedirs(path, exist_ok=True)


def save_image(img, path):
    """
    Save an OpenCV BGR image to disk as a PNG using PIL.
    Args:
        img (np.ndarray): BGR image.
        path (str): File path to save.
    """
    bgr_to_pil(img).save(path)