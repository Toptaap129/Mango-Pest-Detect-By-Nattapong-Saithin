# ============================================================================
# 🥭 MANGO PEST DETECTION - INFERENCE API (FLASK + PYTORCH)
# ============================================================================
# Production-Ready Inference Server compatible with CUDA Training Pipeline
# ============================================================================

import os
import sys
import io
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms

# ============================================================================
# 1. LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api_server.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 2. MODEL ARCHITECTURE (Must match training script exactly)
# ============================================================================
class CUDAOptimizedCNN(nn.Module):
    """
    Architecture must be identical to the training script to load state_dict successfully.
    Source: Pasted_Text_1771252699460.txt
    """
    def __init__(self, num_classes: int, img_size: int = 224):
        super(CUDAOptimizedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        # Weights are loaded from checkpoint, initialization not strictly needed here
        # but kept for completeness if creating new model
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================================
# 3. CONFIGURATION & GLOBAL VARIABLES
# ============================================================================
app = Flask(__name__)
CORS(app)

# Configuration via Environment Variables or Defaults
MODEL_PATH = "best_model.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 96  # Default, will be overridden by checkpoint info

# ImageNet Normalization (Must match training script)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), # Placeholder, will update based on checkpoint
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global Model State
model = None
class_names = []
checkpoint_info = {}

# ============================================================================
# 4. MODEL LOADING UTILITIES
# ============================================================================
def load_model_checkpoint(path: str):
    """
    Loads the inference-ready checkpoint generated by the training script.
    Expects a dictionary containing 'model_state_dict', 'class_names', 'img_size', etc.
    """
    global model, class_names, checkpoint_info, INFERENCE_TRANSFORM
    
    if not os.path.exists(path):
        logger.error(f"❌ Model file not found: {path}")
        raise FileNotFoundError(f"Model file not found: {path}")

    logger.info(f"🚀 Loading model from {path}...")
    logger.info(f"   Device: {DEVICE}")

    try:
        # Load checkpoint dictionary
        # weights_only=False is required because checkpoint contains metadata (lists, strings)
        # Security Note: Ensure model file comes from a trusted source
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        
        # Extract Metadata
        class_names = checkpoint.get('class_names', [])
        img_size = checkpoint.get('img_size', 224)
        num_classes = checkpoint.get('num_classes', len(class_names))
        
        logger.info(f"   Checkpoint Info:")
        logger.info(f"      → Classes: {num_classes}")
        logger.info(f"      → Image Size: {img_size}x{img_size}")
        logger.info(f"      → Training Epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"      → Val Accuracy: {checkpoint.get('val_acc', 'N/A')}")

        # Initialize Model Architecture
        model = CUDAOptimizedCNN(num_classes=num_classes, img_size=img_size)
        
        # Load Weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to Evaluation Mode
        model.to(DEVICE)
        model.eval()

        # Update Transform to match exact image size from checkpoint
        INFERENCE_TRANSFORM = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        checkpoint_info = {
            'path': path,
            'loaded_at': datetime.now().isoformat(),
            'architecture': checkpoint.get('model_architecture', 'CUDAOptimizedCNN'),
            'pytorch_version': checkpoint.get('pytorch_version', 'Unknown')
        }

        logger.info(f"✅ Model loaded successfully.")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise e

# ============================================================================
# 5. API ENDPOINTS
# ============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts an image file and returns prediction probabilities.
    """
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 503

    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        # Read and Process Image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply Transform (Resize, ToTensor, Normalize)
        input_tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
        
        inference_time = time.time() - start_time
        
        # Parse Results
        probs_np = probabilities.cpu().numpy()[0]
        pred_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[pred_idx])
        
        # Build Response
        # Ensure we don't exceed class names list length
        safe_classes = class_names if len(class_names) >= len(probs_np) else class_names + [f"Class_{i}" for i in range(len(class_names), len(probs_np))]
        
        predictions_list = [
            {'class': safe_classes[i], 'probability': round(float(p), 4)}
            for i, p in enumerate(probs_np)
        ]
        # Sort by probability descending
        predictions_list.sort(key=lambda x: x['probability'], reverse=True)

        response = {
            'success': True,
            'prediction': {
                'class': safe_classes[pred_idx],
                'confidence': round(confidence, 4),
                'index': pred_idx
            },
            'all_probabilities': predictions_list,
            'inference_time_ms': round(inference_time * 1000, 2),
            'model_info': {
                'path': os.path.basename(MODEL_PATH),
                'device': str(DEVICE)
            }
        }

        logger.info(f"🔮 Prediction: {response['prediction']['class']} ({confidence:.2%}) in {inference_time*1000:.1f}ms")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Internal server error during prediction',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """
    System health check endpoint.
    """
    is_model_loaded = model is not None
    return jsonify({
        'status': 'healthy' if is_model_loaded else 'degraded',
        'service': 'Mango Pest Detection API',
        'framework': 'PyTorch',
        'version': torch.__version__,
        'device': str(DEVICE),
        'cuda_available': torch.cuda.is_available(),
        'model_loaded': is_model_loaded,
        'model_info': checkpoint_info if is_model_loaded else None,
        'num_classes': len(class_names),
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/classes', methods=['GET'])
def get_classes():
    """
    Returns list of all known disease classes.
    """
    return jsonify({
        'num_classes': len(class_names),
        'class_names': class_names
    }), 200

# ============================================================================
# 6. MAIN ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    logger.info("="*80)
    logger.info("🥭 MANGO PEST DETECTION API SERVER STARTING")
    logger.info("="*80)
    
    try:
        load_model_checkpoint(MODEL_PATH)
        
        logger.info("-"*80)
        logger.info("🌍 Server Configuration:")
        logger.info(f"   Host: 0.0.0.0")
        logger.info(f"   Port: 5000")
        logger.info(f"   Debug: False")
        logger.info(f"   Model: {MODEL_PATH}")
        logger.info("-"*80)
        logger.info("✅ Server Ready to Accept Requests")
        logger.info("="*80)
        
        # Use threaded=True for handling concurrent requests
        # In production, use Gunicorn/uWSGI instead of app.run()
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        logger.critical(f"🚫 Failed to start server: {e}")
        sys.exit(1)

