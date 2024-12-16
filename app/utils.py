import os
from typing import List
import cv2
import numpy as np
import time
import shutil
from fpdf import FPDF
from config import Config
from pathlib import Path
from werkzeug.utils import secure_filename
import logging

# Configure logger
logger = logging.getLogger(__name__)

def allowed_file(filename: str) -> bool:
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """Clean up old temporary files"""
    try:
        current_time = time.time()
        directory_path = Path(directory)
        
        if not directory_path.exists():
            return
            
        for file_path in directory_path.glob('*'):
            if file_path.is_file():
                if file_path.stat().st_mtime < current_time - (max_age_hours * 3600):
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete file {file_path}: {e}")
                        
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")

def validate_file(file):
    """Validate uploaded file"""
    if not file:
        return False, "No file provided"
        
    if not file.filename:
        return False, "No filename provided"
        
    if not allowed_file(file.filename):
        return False, "File type not allowed"
        
    return True, None

def secure_save_file(file, upload_folder):
    """Securely save uploaded file"""
    try:
        filename = secure_filename(file.filename)
        filepath = Path(upload_folder) / filename
        
        # Ensure upload directory exists and has correct permissions
        upload_dir = Path(upload_folder)
        upload_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(upload_dir, 0o755)
        
        file.save(str(filepath))
        os.chmod(filepath, 0o644)  # Set file permissions
        
        return str(filepath)
    except Exception as e:
        raise Exception(f"Failed to save file: {str(e)}")