import numpy as np;
import cv2

def gamma_correction(image, gamma=1.0):
    # Apply gamma correction
    gamma_corrected = np.uint8(np.clip(((image / 255.0) ** gamma) * 255.0, 0, 255))
    return gamma_corrected