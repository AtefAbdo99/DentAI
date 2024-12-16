from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from model_handler import ModelHandler
from diagnosis_data import DiagnosisData
from config import Config
import logging
from datetime import datetime
import utils

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = Path(Config.UPLOAD_FOLDER)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, 
    template_folder=str(Path(__file__).parent.parent / 'templates'),
    static_folder=str(Path(__file__).parent.parent / 'static')
)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config.from_object(Config)
CORS(app)

# Configure logging with proper Linux paths
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(Path('/var/log/dentai.log') if os.path.exists('/var/log') else Path('dentai.log'))),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize model with proper error handling
try:
    model_handler = ModelHandler(Config.MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model_handler = ModelHandler.get_dummy_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        # Validate file
        is_valid, error_msg = utils.validate_file(request.files.get('file'))
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        file = request.files['file']
        try:
            # Securely save file with proper permissions
            filepath = utils.secure_save_file(file, app.config['UPLOAD_FOLDER'])
            
            # Process image
            image_tensor = model_handler.preprocess_image(filepath)
            if image_tensor is None:
                return jsonify({'error': 'Failed to process image'}), 400
                
            predictions = model_handler.predict(image_tensor)
            
            # Get relative path for client
            relative_path = os.path.relpath(filepath, app.config['UPLOAD_FOLDER'])
            
            response_data = {
                'predictions': predictions,
                'findings': DiagnosisData.FINDINGS_MAP.get(predictions[0][0].lower(), []),
                'recommendations': DiagnosisData.RECOMMENDATIONS_MAP.get(predictions[0][0].lower(), []),
                'management': DiagnosisData.MANAGEMENT_MAP.get(predictions[0][0].lower(), {}),
                'image_path': f'/uploads/{relative_path}'
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': 'Failed to process image'}), 500
            
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)