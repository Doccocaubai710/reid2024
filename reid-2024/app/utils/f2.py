import torch
import torch.onnx
from torchreid.models.lmbn_n import LMBN_n
def convert_pth_to_onnx(model, model_path, onnx_path, dummy_input):
    """
    Convert PyTorch .pth model to ONNX format
    """
    # Load pretrained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Remove 'model.' prefix if it exists
    if 'model.backone.0.conv.weight' in checkpoint:
        new_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                new_checkpoint[new_key] = value
            else:
                new_checkpoint[key] = value
        checkpoint = new_checkpoint
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    
    print(f"Converted to ONNX: {onnx_path}")

# Usage example for LMBN_n:
import argparse

# Create args object with required attributes
args = argparse.Namespace()
args.num_classes = 767 # Adjust to your dataset's number of classes
args.feats = 512         # Feature dimension
args.activation_map = False  # Set to False for inference

model = LMBN_n(args)  # Initialize your model
dummy_input = torch.randn(1, 3, 256, 128)  # Standard ReID input size
convert_pth_to_onnx(model, "/home/aidev/workspace/reid/Thesis/reid-2024/app/assets/models/lmbn_n_cuhk03_d.pth", "model.onnx", dummy_input)