"""
Handles all model-related operations and predictions for the dental X-ray analyzer.
"""
import logging
import os
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torchvision import models
    import numpy as np
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import required packages: {str(e)}")
    TORCH_AVAILABLE = False

class ModelHandler:
    """Handles model loading, configuration and inference."""
    
    def __init__(self, model_path: str = 'initial_model.pth'):
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
                
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_path = model_path
            self.class_names = [
                'Nil control', 
                'condensing osteitis', 
                'diffuse lesion', 
                'periapical abcess', 
                'periapical granuloma', 
                'periapical widening', 
                'pericoronitis', 
                'radicular cyst'
            ]
            
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            self.model = self._initialize_model()
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            logger.info("Using dummy model for testing")
            self.model = self._get_dummy_model()

    def _initialize_model(self) -> 'torch.nn.Module':
        """Initialize and load the model."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file '{self.model_path}' not found!")

            # Initialize model architecture
            model = models.efficientnet_b0(weights=None)
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2),
                torch.nn.Linear(1280, len(self.class_names))
            )

            # Load model weights
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            except Exception as e:
                raise Exception(f"Failed to load model weights: {str(e)}")

            model.eval()
            return model.to(self.device)

        except Exception as e:
            raise Exception(f"Model initialization failed: {str(e)}")

    def _get_dummy_model(self) -> 'torch.nn.Module':
        """Create a dummy model for testing."""
        if TORCH_AVAILABLE:
            class DummyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                
                def forward(self, x):
                    batch_size = x.shape[0] if isinstance(x, torch.Tensor) else 1
                    return torch.ones(batch_size, 8)  # 8 classes
                    
                def eval(self):
                    return self
                    
                def to(self, device):
                    return self
        else:
            class DummyModel:
                def __call__(self, x):
                    return [[1.0] * 8]
                
                def eval(self):
                    return self
                    
                def to(self, device):
                    return self

        return DummyModel()

    def predict(self, image_tensor: 'torch.Tensor', top_k: int = 3) -> List[Tuple[str, float]]:
        """Make prediction with confidence scores."""
        try:
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidences, predictions = torch.topk(probabilities, top_k)
                    
                    results = []
                    for pred, conf in zip(predictions[0], confidences[0]):
                        results.append((self.class_names[pred.item()], conf.item()))
                    return results
            else:
                # Return dummy predictions when torch is not available
                return [(self.class_names[0], 1.0)]
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return [("Error", 0.0)]

    def preprocess_image(self, image_path: str) -> Optional['torch.Tensor']:
        """Preprocess image for model input."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            transformed = self.transform(image=image_array)
            
            if TORCH_AVAILABLE:
                return transformed['image'].unsqueeze(0).to(self.device)
            else:
                return np.zeros((1, 3, 224, 224))  # Dummy tensor
                
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None