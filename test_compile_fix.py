#!/usr/bin/env python3
"""Test script to verify the compilation fix works."""

import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.model import TransformerLM


def test_compile_fix():
    """Test that the model can be compiled and used with gradient checkpointing."""
    print("🧪 Testing compilation fix...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, testing on CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"✅ Testing on {torch.cuda.get_device_name(0)}")
    
    # Create a smaller model for testing
    model = TransformerLM(
        vocab_size=50257,
        max_seq_len=512,
        dim=512,
        n_layers=6,
        n_heads=8,
        head_dim=64,
        intermediate_size=2048,
        dropout=0.0,
        tie_embeddings=True,
        use_flash=False,  # Disable for compatibility testing
        use_gradient_checkpointing=True,  # Enable checkpointing
    )
    
    model = model.to(device)
    
    # Test normal forward pass first
    print("Testing normal forward pass...")
    batch_size = 2
    seq_len = 256
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    
    # Normal forward pass
    outputs = model(input_ids, labels=labels)
    loss1 = outputs['loss']
    print(f"✅ Normal forward pass successful. Loss: {loss1.item():.4f}")
    
    # Test backward pass
    print("Testing backward pass...")
    loss1.backward()
    print("✅ Normal backward pass successful")
    
    # Clear gradients
    model.zero_grad()
    
    # Now test with compilation
    print("Testing with torch.compile()...")
    try:
        compiled_model = torch.compile(model, mode="default")
        
        # Warmup
        for i in range(3):
            with torch.no_grad():
                _ = compiled_model(input_ids)
                
        print("✅ Compilation warmup successful")
        
        # Test compiled forward pass
        outputs = compiled_model(input_ids, labels=labels)
        loss2 = outputs['loss']
        print(f"✅ Compiled forward pass successful. Loss: {loss2.item():.4f}")
        
        # Test compiled backward pass
        loss2.backward()
        print("✅ Compiled backward pass successful")
        
        print(f"\n🎉 All tests passed!")
        print(f"Loss difference: {abs(loss1.item() - loss2.item()):.6f}")
        
    except Exception as e:
        print(f"❌ Compilation test failed: {e}")
        return False
        
    return True


def test_rope_stateless():
    """Test that RoPE is now stateless."""
    print("\n🧪 Testing RoPE stateless behavior...")
    
    from cs336_basics.model.components import RotaryPositionEmbedding
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rope = RotaryPositionEmbedding(dim=64).to(device)
    
    # Create test tensors
    batch_size = 2
    seq_len = 128
    n_heads = 8
    head_dim = 64
    
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
    
    # Test multiple calls
    q1, k1 = rope(q, k)
    q2, k2 = rope(q, k)
    
    # Results should be identical (stateless)
    assert torch.allclose(q1, q2), "RoPE is not stateless!"
    assert torch.allclose(k1, k2), "RoPE is not stateless!"
    
    print("✅ RoPE is stateless - multiple calls produce identical results")
    
    # Test with compilation
    try:
        compiled_rope = torch.compile(rope)
        q3, k3 = compiled_rope(q, k)
        
        # Should still match
        assert torch.allclose(q1, q3, atol=1e-5), "Compiled RoPE doesn't match!"
        assert torch.allclose(k1, k3, atol=1e-5), "Compiled RoPE doesn't match!"
        
        print("✅ Compiled RoPE produces consistent results")
        
    except Exception as e:
        print(f"❌ RoPE compilation failed: {e}")
        return False
        
    return True


if __name__ == "__main__":
    print("🚀 Testing compilation fixes")
    print("=" * 50)
    
    success1 = test_rope_stateless()
    success2 = test_compile_fix()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All compilation fixes verified!")
        print("✅ Ready for optimized training")
    else:
        print("❌ Some tests failed")
        sys.exit(1)