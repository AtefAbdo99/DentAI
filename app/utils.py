import os
from typing import List
import cv2
import numpy as np
import time
import shutil
from fpdf import FPDF
from config import Config

def allowed_file(filename: str) -> bool:
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """Clean up old temporary files"""
    current_time = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.getmtime(filepath) < current_time - (max_age_hours * 3600):
            os.remove(filepath)