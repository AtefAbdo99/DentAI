import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Model settings
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, os.getenv('MODEL_FILE', 'initial_model.pth'))
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
    DEBUG = os.getenv('FLASK_ENV') == 'development'
    
    # Font settings
    FONT_DIR = os.path.join(BASE_DIR, 'fonts')
    
    @classmethod
    def init_app(cls):
        """Initialize application directories"""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.FONT_DIR, exist_ok=True)