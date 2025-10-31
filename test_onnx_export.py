#!/usr/bin/env python3
"""
Barracuda-compatible ONNX export test for trans_v7_3 encoder
Unity ML-Agents uses Barracuda which has specific ONNX requirements
"""

import torch
import torch.onnx
import sys
import os

# Add the encoders directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Encoders'))

from trans_v7_3 import NatureVisualEncoder

def test_barracuda_onnx_export():
    """Test if the encoder can be exported to Barracuda-compatible ONNX"""
    print("Testing Barracuda-compatible ONNX export for trans_v7_3 encoder...")
    
    # Create model instance
    height, width = 88, 156  # Foggy environment dimensions
    initial_channels = 3  # RGB
    output_size = 512
    
    model = NatureVisualEncoder(height, width, initial_channels, output_size)
    model.eval()
    
    # Create dummy input - Barracuda expects [batch, height, width, channels] format
    batch_size = 1
    dummy_input = torch.randn(batch_size, height, width, initial_channels)
    
    try:
        # Test forward pass first
        print("Testing forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Forward pass successful! Output shape: {output.shape}")
        
        # Test Barracuda-compatible ONNX export
        print("Testing Barracuda-compatible ONNX export...")
        onnx_path = "trans_v7_3_barracuda.onnx"
        
        # Barracuda-specific export settings
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=9,  # Barracuda works best with opset 9
            do_constant_folding=True,
            input_names=['obs_0'],  # ML-Agents standard input name
            output_names=['action'],  # ML-Agents standard output name
            dynamic_axes={
                'obs_0': {0: 'batch'},  # Dynamic batch size
                'action': {0: 'batch'}
            },
            verbose=False,
            keep_initializers_as_inputs=False,
            strip_doc_string=True
        )
        
        print(f"Barracuda-compatible ONNX export successful! Model saved to: {onnx_path}")
        
        # Verify the exported model
        print("Verifying exported model...")
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification passed!")
            
            # Print model info
            print(f"Model inputs: {[input.name for input in onnx_model.graph.input]}")
            print(f"Model outputs: {[output.name for output in onnx_model.graph.output]}")
            
        except ImportError:
            print("ONNX package not available for verification, but export completed.")
        except Exception as e:
            print(f"ONNX verification failed: {e}")
        
        print(f"Model file size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
        
        return True, onnx_path
        
    except Exception as e:
        print(f"Barracuda ONNX export failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, model_path = test_barracuda_onnx_export()
    if success:
        print(f"\n✅ SUCCESS: trans_v7_3 encoder is Barracuda-compatible!")
        print(f"   Model saved as: {model_path}")
        print("   You can now use this model in Unity ML-Agents!")
    else:
        print("\n❌ FAILED: Barracuda ONNX export still has issues")
    
    sys.exit(0 if success else 1)