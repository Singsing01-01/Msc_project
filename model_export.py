import torch
import torch.onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from model_a_gnn import ModelA_GNN
from model_b_similarity import ModelB_Similarity


class ModelExporter:
    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def export_to_onnx(self, 
                      model: torch.nn.Module,
                      model_name: str,
                      export_path: str,
                      input_shape: Tuple[int, ...] = (1, 1, 64, 64),
                      max_nodes: int = 350) -> str:
        
        model.eval()
        
        dummy_image = torch.randn(*input_shape).to(self.device)
        dummy_mask = torch.ones(input_shape[0], max_nodes, dtype=torch.bool).to(self.device)
        
        input_names = ['images', 'node_masks']
        output_names = ['predicted_coords', 'adjacency_matrix', 'node_counts']
        
        dynamic_axes = {
            'images': {0: 'batch_size'},
            'node_masks': {0: 'batch_size'},
            'predicted_coords': {0: 'batch_size'},
            'adjacency_matrix': {0: 'batch_size'},
            'node_counts': {0: 'batch_size'}
        }
        
        onnx_path = os.path.join(export_path, f"{model_name}.onnx")
        os.makedirs(export_path, exist_ok=True)
        
        try:
            torch.onnx.export(
                model,
                (dummy_image, dummy_mask),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            print(f"Successfully exported {model_name} to {onnx_path}")
            return onnx_path
            
        except Exception as e:
            print(f"Failed to export {model_name} to ONNX: {e}")
            return None
    
    def verify_onnx_model(self, 
                         onnx_path: str,
                         pytorch_model: torch.nn.Module,
                         input_shape: Tuple[int, ...] = (1, 1, 64, 64),
                         max_nodes: int = 350) -> bool:
        
        try:
            ort_session = ort.InferenceSession(onnx_path)
            
            test_image = torch.randn(*input_shape)
            test_mask = torch.ones(input_shape[0], max_nodes, dtype=torch.bool)
            
            pytorch_model.eval()
            pytorch_model.to('cpu')
            
            with torch.no_grad():
                pytorch_output = pytorch_model(test_image, test_mask)
            
            onnx_inputs = {
                'images': test_image.numpy().astype(np.float32),
                'node_masks': test_mask.numpy()
            }
            
            onnx_outputs = ort_session.run(None, onnx_inputs)
            
            pytorch_coords = pytorch_output['predicted_coords'].numpy()
            pytorch_adj = pytorch_output['adjacency_matrix'].numpy()
            
            onnx_coords = onnx_outputs[0]
            onnx_adj = onnx_outputs[1]
            
            coords_diff = np.abs(pytorch_coords - onnx_coords).max()
            adj_diff = np.abs(pytorch_adj - onnx_adj).max()
            
            print(f"Verification results:")
            print(f"  Max coordinate difference: {coords_diff:.6f}")
            print(f"  Max adjacency difference: {adj_diff:.6f}")
            
            tolerance = 1e-4
            if coords_diff < tolerance and adj_diff < tolerance:
                print("[SUCCESS] ONNX model verification passed!")
                return True
            else:
                print("[ERROR] ONNX model verification failed - outputs differ significantly")
                return False
                
        except Exception as e:
            print(f"ONNX verification failed: {e}")
            return False
    
    def export_models_with_verification(self, 
                                      models: Dict[str, torch.nn.Module],
                                      export_dir: str) -> Dict[str, str]:
        
        print("Exporting models to ONNX format...")
        
        exported_paths = {}
        
        for model_name, model in models.items():
            print(f"\nExporting {model_name}...")
            
            onnx_path = self.export_to_onnx(
                model, model_name, export_dir
            )
            
            if onnx_path:
                print(f"Verifying {model_name}...")
                if self.verify_onnx_model(onnx_path, model):
                    exported_paths[model_name] = onnx_path
                else:
                    print(f"Verification failed for {model_name}")
        
        return exported_paths


class ONNXInferenceEngine:
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path)
        
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Loaded ONNX model from {onnx_path}")
        print(f"Input names: {self.input_names}")
        print(f"Output names: {self.output_names}")
    
    def predict(self, 
               image: np.ndarray,
               node_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        
        if len(image.shape) == 2:
            image = image.reshape(1, 1, *image.shape)
        elif len(image.shape) == 3:
            image = image.reshape(1, *image.shape)
        
        if node_mask is None:
            node_mask = np.ones((image.shape[0], 350), dtype=bool)
        elif len(node_mask.shape) == 1:
            node_mask = node_mask.reshape(1, -1)
        
        inputs = {
            'images': image.astype(np.float32),
            'node_masks': node_mask
        }
        
        outputs = self.session.run(None, inputs)
        
        return {
            'predicted_coords': outputs[0],
            'adjacency_matrix': outputs[1],
            'node_counts': outputs[2] if len(outputs) > 2 else None
        }
    
    def batch_predict(self, 
                     images: List[np.ndarray],
                     node_masks: Optional[List[np.ndarray]] = None) -> List[Dict[str, np.ndarray]]:
        
        results = []
        
        for i, image in enumerate(images):
            mask = node_masks[i] if node_masks else None
            result = self.predict(image, mask)
            results.append(result)
        
        return results


def create_deployment_package(models: Dict[str, torch.nn.Module],
                            export_dir: str,
                            include_requirements: bool = True) -> str:
    
    print("Creating deployment package...")
    
    os.makedirs(export_dir, exist_ok=True)
    
    exporter = ModelExporter()
    exported_paths = exporter.export_models_with_verification(models, export_dir)
    
    demo_script = '''
import numpy as np
from model_export import ONNXInferenceEngine

def demo_inference():
    """
    Demo script showing how to use exported ONNX models
    """
    
    # Load ONNX model
    model_path = "ModelB_Similarity.onnx"  # Change to your model path
    engine = ONNXInferenceEngine(model_path)
    
    # Create dummy input (replace with your actual image)
    dummy_image = np.random.rand(64, 64).astype(np.float32)
    
    # Run inference
    results = engine.predict(dummy_image)
    
    print("Inference results:")
    print(f"Predicted coordinates shape: {results['predicted_coords'].shape}")
    print(f"Adjacency matrix shape: {results['adjacency_matrix'].shape}")
    
    return results

if __name__ == "__main__":
    demo_inference()
'''
    
    with open(os.path.join(export_dir, 'demo_inference.py'), 'w') as f:
        f.write(demo_script)
    
    if include_requirements:
        requirements = '''
onnxruntime>=1.12.0
numpy>=1.21.0
torch>=1.12.0
'''
        with open(os.path.join(export_dir, 'requirements.txt'), 'w') as f:
            f.write(requirements)
    
    readme = f'''
# Image-to-Graph Models Deployment Package

## Overview
This package contains exported ONNX models for image-to-graph prediction.

## Exported Models
{chr(10).join([f"- {name}: {path}" for name, path in exported_paths.items()])}

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Basic Inference
```python
from model_export import ONNXInferenceEngine

# Load model
engine = ONNXInferenceEngine("ModelB_Similarity.onnx")

# Predict on image
image = np.random.rand(64, 64).astype(np.float32)
results = engine.predict(image)

# Access results
coords = results['predicted_coords']
adjacency = results['adjacency_matrix']
```

### Demo
```bash
python demo_inference.py
```

## Model Specifications
- Input: 64x64 grayscale images
- Output: Node coordinates + adjacency matrix
- Format: ONNX (compatible with most deployment frameworks)
'''
    
    with open(os.path.join(export_dir, 'README.md'), 'w') as f:
        f.write(readme)
    
    print(f"Deployment package created in: {export_dir}")
    print(f"Exported models: {list(exported_paths.keys())}")
    
    return export_dir


def test_model_export():
    print("Testing model export functionality...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {
        'ModelB_Similarity_Test': ModelB_Similarity().to(device)
    }
    
    export_dir = "/Users/jeremyfang/Downloads/image_to_graph/exported_models"
    
    deployment_dir = create_deployment_package(models, export_dir)
    
    onnx_path = os.path.join(export_dir, "ModelB_Similarity_Test.onnx")
    if os.path.exists(onnx_path):
        print(f"\nTesting ONNX inference engine...")
        engine = ONNXInferenceEngine(onnx_path)
        
        test_image = np.random.rand(64, 64).astype(np.float32)
        results = engine.predict(test_image)
        
        print(f"Inference successful!")
        print(f"Coordinates shape: {results['predicted_coords'].shape}")
        print(f"Adjacency shape: {results['adjacency_matrix'].shape}")
    
    print("Model export test completed!")


if __name__ == "__main__":
    test_model_export()