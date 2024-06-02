import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import torch
import logging

app = Flask(__name__)

# Load your YOLOv8 model
model_path = "C:/Users/achsu/Downloads/best.pt"  # Update this to your model path
model = YOLO(model_path)

# Configure logging
logging.basicConfig(filename='flask_server.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

@app.route('/')
def home():
    return "YOLOv8 Model Hosting with Flask"

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received a request on /predict")
    if 'image' not in request.files:
        logging.error("No image provided")
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    logging.info("Image received: %s", image_file.filename)

    try:
        image = Image.open(image_file.stream).convert('RGB')
        # Convert image to tensor
        img_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Adjust shape if necessary
        logging.info("Image tensor shape: %s", img_tensor.shape)
        
        # Run inference
        results = model(img_tensor)
        logging.info("Inference results: %s", results)
        
        # Parse the results
        predictions = []
        for result in results:
            for det in result:
                x_min, y_min, x_max, y_max, conf, cls = det
                predictions.append({
                    'x_min': x_min.item(),
                    'y_min': y_min.item(),
                    'x_max': x_max.item(),
                    'y_max': y_max.item(),
                    'confidence': conf.item(),
                    'class': int(cls.item())
                })

        logging.info("Parsed predictions: %s", predictions)
        return jsonify({'predictions': predictions})

    except Exception as e:
        logging.error("Error processing image: %s", str(e))
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
