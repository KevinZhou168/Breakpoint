"""One input preparation path for the UI and CLI."""
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_input(source):
    with Image.open(source) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("Please choose an image under 20 megapixels")
        image = ImageOps.exif_transpose(image).convert("RGB")
        pixels = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if max(pixels.shape[:2]) > 960:
        scale = 960 / max(pixels.shape[:2])
        pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return pixels
