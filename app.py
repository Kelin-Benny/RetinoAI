from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from pathlib import Path
import torch

# Import our modules
from .inference import RetinoAIInference
from .gradcam import get_grad_cam
from .report import generate_pdf_report
from .configs import OCT_CLASSES, FUNDUS_CLASSES

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Initialize inference engine
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize the inference engine (we'll create a simple version for now)

# Initialize the inference engine
inference_engine = RetinoAIInference(device=device)

@app.route('/')
def index():
    return render_template('index.html', oct_classes=OCT_CLASSES, fundus_classes=FUNDUS_CLASSES)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Read image
    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'})

    # Collect patient metadata
    patient_data = {
        'age': request.form.get('age', 'Not provided'),
        'diabetic': request.form.get('diabetic', 'Not provided'),
        'family_history': request.form.get('family_history', 'Not provided')
    }

    # Return mock data for frontend demo
    mock_result = {
        'modality': 'OCT',
        'modality_confidence': 95.7,
        'diagnosis': 'CNV',
        'confidence': 87.3,
        'probabilities': {
            'CNV': 87.3,
            'DME': 8.2,
            'DRUSEN': 2.1,
            'NORMAL': 1.2,
            'AMD': 0.8,
            'RVO': 0.3,
            'CSC': 0.1,
            'ERM': 0.0
        },
        'patient_data': patient_data
    }
    return jsonify(mock_result)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'})
    
    files = request.files.getlist('files')
    results = []
    
    for file in files:
        try:
            # Mock result for each file
            mock_result = {
                'filename': file.filename,
                'modality': 'Fundus' if 'fundus' in file.filename.lower() else 'OCT',
                'diagnosis': 'Diabetic Retinopathy' if 'fundus' in file.filename.lower() else 'CNV',
                'confidence': 92.5,
                'error': None
            }
            results.append(mock_result)
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return jsonify({'results': results})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    pdf_buffer = generate_pdf_report(data)
    
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name='retinoai_report.pdf',
        mimetype='application/pdf'
    )

def create_app():
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    return app

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app on all available interfaces
    app.run(debug=True, host='0.0.0.0', port=5000)