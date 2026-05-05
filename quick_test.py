from PIL import Image
from model import ImageProcessor
from model import PatchEmbedding
from model import TransformerEncoder
import yaml
import torch

if __name__ == "__main__":
    # Parameters
    batch_size = 8
    seq_length = 64    # e.g., 8x8 patches
    embedding_dim = 128
    depth = 4
    heads = 8
    mlp_dim = 256

    # 1. Create a dummy input tensor [Batch, Sequence, Features]
    x = torch.randn(batch_size, seq_length, embedding_dim)
    print(f"Input shape: {x.shape}")

    # 2. Initialize your Transformer
    model = TransformerEncoder(
        depth=depth, 
        dim=embedding_dim, 
        n_heads=heads, 
        dropout_rate=0.1, 
        hidden_dim=mlp_dim
    )

    # 3. Pass data through
    try:
        output = model(x)
        print(f"Output shape: {output.shape}")
        
        # 4. Verify shapes match
        if x.shape == output.shape:
            print("✅ Shape Test Passed! The Transformer preserved the dimensions.")
        else:
            print("❌ Shape Test Failed! Input and Output dimensions differ.")
            
    except Exception as e:
        print(f"❌ Execution Failed with error: {e}")

    output.mean().backward()
    print("✅ Backward pass successful!")