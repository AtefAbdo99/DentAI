import os
from pathlib import Path
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class Config:
    BASE_DIR = Path(__file__).parent
    
    # Model settings
    MODEL_DIR = BASE_DIR / 'models'
    MODEL_PATH = MODEL_DIR / os.getenv('MODEL_FILE', 'initial_model.pth')
    
    # Upload settings
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
    DEBUG = os.getenv('FLASK_ENV') == 'development'
    
    @classmethod
    def init_app(cls):
        """Initialize application directories with proper permissions"""
        try:
            for directory in [cls.MODEL_DIR, cls.UPLOAD_FOLDER]:
                directory.mkdir(parents=True, exist_ok=True)
                # Set directory permissions to 755
                os.chmod(directory, 0o755)
                
            # Create .htaccess to prevent direct file access
            htaccess = cls.UPLOAD_FOLDER / '.htaccess'
            htaccess.write_text('Deny from all')
            
        except Exception as e:
            logger.error(f"Failed to initialize directories: {e}")
            raise