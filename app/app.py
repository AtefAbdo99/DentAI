from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from app.model_handler import ModelHandler
from app.diagnosis_data import DiagnosisData
from app.config import Config
import logging
from datetime import datetime

app = Flask(__name__, 
    template_folder='../templates',
    static_folder='../static'
)
app.config.from_object(Config)
Config.init_app()
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
try:
    model_handler = ModelHandler(Config.MODEL_PATH)
except FileNotFoundError:
    logger.warning(f"Model file not found at {Config.MODEL_PATH}. Using dummy model for testing.")
    model_handler = ModelHandler.get_dummy_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image
            image_tensor = model_handler.preprocess_image(filepath)
            if image_tensor is None:
                return jsonify({'error': 'Failed to process image'}), 400
                
            predictions = model_handler.predict(image_tensor)
            
            # Prepare response data
            response_data = {
                'predictions': predictions,
                'findings': DiagnosisData.FINDINGS_MAP.get(predictions[0][0].lower(), []),
                'recommendations': DiagnosisData.RECOMMENDATIONS_MAP.get(predictions[0][0].lower(), []),
                'management': DiagnosisData.MANAGEMENT_MAP.get(predictions[0][0].lower(), {}),
                'image_path': filepath
            }
            
            return jsonify(response_data)
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)