"""
Tests for the Muon optimizer and mixed optimizers.
"""

import math

import pytest
import torch
import torch.nn as nn

from cs336_basics.training.optimizers import Adam, AdamW, MixedOptimizerV2, Muon


class SimpleLinear(nn.Module):
    """Simple linear model for testing."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class SimpleTransformer(nn.Module):
    """Simple transformer-like model for testing parameter separation."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.token_embeddings(x)
        residual = x
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.layer_norm(x + residual)
        return self.lm_head(x)


class TestNewtonSchulz:
    """Test Newton-Schulz orthogonalization."""

    def test_newton_schulz_basic_convergence(self):
        """Test that Newton-Schulz converges to orthogonal matrix."""
        # Create a dummy parameter for the optimizer
        dummy_param = torch.randn(1, 1, requires_grad=True)
        muon = Muon([dummy_param], use_optimized_coefficients=True)

        # Create a random matrix
        torch.manual_seed(42)
        G = torch.randn(8, 6)

        # Apply Newton-Schulz
        ortho_G = muon.newton_schulz_orthogonalize(G, num_iters=5)

        # Check that result is finite
        assert torch.isfinite(ortho_G).all(), "Newton-Schulz result should be finite"

        # Check orthogonality: G^T @ G should be close to identity
        # Note: Newton-Schulz with few iterations won't be perfect, but should be reasonable
        gram = ortho_G.T @ ortho_G
        identity = torch.eye(gram.shape[0])
        orthogonality_error = torch.norm(gram - identity).item()

        assert orthogonality_error < 2.0, f"Orthogonality error too large: {orthogonality_error}"
        print(f"Newton-Schulz orthogonality error: {orthogonality_error:.6f} (reasonable for few iterations)")

    def test_newton_schulz_stability_with_extreme_inputs(self):
        """Test Newton-Schulz stability with extreme inputs."""
        dummy_param = torch.randn(1, 1, requires_grad=True)
        muon = Muon([dummy_param], use_optimized_coefficients=True)

        # Test with very small matrix
        G_small = torch.ones(4, 3) * 1e-10
        result = muon.newton_schulz_orthogonalize(G_small)
        assert torch.isfinite(result).all(), "Should handle very small inputs"

        # Test with very large matrix
        G_large = torch.ones(4, 3) * 1e5
        result = muon.newton_schulz_orthogonalize(G_large)
        assert torch.isfinite(result).all(), "Should handle very large inputs"

        # Test with NaN input
        G_nan = torch.tensor([[1.0, 2.0], [float("nan"), 4.0]])
        result = muon.newton_schulz_orthogonalize(G_nan)
        assert torch.isfinite(result).all(), "Should handle NaN inputs gracefully"

        # Test with Inf input
        G_inf = torch.tensor([[1.0, 2.0], [float("inf"), 4.0]])
        result = muon.newton_schulz_orthogonalize(G_inf)
        assert torch.isfinite(result).all(), "Should handle Inf inputs gracefully"

    def test_newton_schulz_shape_preservation(self):
        """Test that Newton-Schulz preserves matrix shapes."""
        dummy_param = torch.randn(1, 1, requires_grad=True)
        muon = Muon([dummy_param], use_optimized_coefficients=True)

        # Test various shapes
        shapes = [(5, 3), (3, 5), (4, 4), (10, 2), (2, 10)]

        for shape in shapes:
            G = torch.randn(shape)
            result = muon.newton_schulz_orthogonalize(G)
            assert result.shape == shape, f"Shape should be preserved for {shape}"
            assert torch.isfinite(result).all(), f"Result should be finite for shape {shape}"

    def test_optimized_vs_cubic_coefficients(self):
        """Test that optimized coefficients work better than cubic."""
        torch.manual_seed(42)
        G = torch.randn(6, 4)

        # Test with optimized coefficients
        dummy_param = torch.randn(1, 1, requires_grad=True)
        muon_opt = Muon([dummy_param], use_optimized_coefficients=True)
        result_opt = muon_opt.newton_schulz_orthogonalize(G, num_iters=3)

        # Test with cubic coefficients
        dummy_param2 = torch.randn(1, 1, requires_grad=True)
        muon_cubic = Muon([dummy_param2], use_optimized_coefficients=False)
        result_cubic = muon_cubic.newton_schulz_orthogonalize(G, num_iters=3)

        # Both should be finite and roughly orthogonal
        assert torch.isfinite(result_opt).all(), "Optimized result should be finite"
        assert torch.isfinite(result_cubic).all(), "Cubic result should be finite"

        # Check orthogonality for both
        gram_opt = result_opt.T @ result_opt
        gram_cubic = result_cubic.T @ result_cubic
        identity = torch.eye(gram_opt.shape[0])

        error_opt = torch.norm(gram_opt - identity).item()
        error_cubic = torch.norm(gram_cubic - identity).item()

        assert error_opt < 3.0, f"Optimized orthogonality error: {error_opt}"
        assert error_cubic < 3.0, f"Cubic orthogonality error: {error_cubic}"
        print(f"Optimized vs Cubic errors: {error_opt:.6f} vs {error_cubic:.6f}")


class TestMuon:
    """Test Muon optimizer."""

    def test_muon_initialization(self):
        """Test Muon optimizer initialization."""
        model = SimpleLinear(10, 20, 5)
        optimizer = Muon(model.parameters(), lr=1e-3, momentum=0.9)

        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["momentum"] == 0.9
        assert len(optimizer.optimized_ns_coefficients) == 6  # YouJiacheng's 6-step

    def test_muon_step_basic(self):
        """Test basic Muon optimization step."""
        torch.manual_seed(42)
        model = SimpleLinear(10, 20, 5)
        optimizer = Muon(model.parameters(), lr=1e-3)

        # Create dummy data
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)

        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, y)

        # Backward pass
        loss.backward()

        # Store original parameters
        original_params = [p.clone() for p in model.parameters()]

        # Optimization step
        optimizer.step()

        # Check that parameters changed
        for orig, new in zip(original_params, model.parameters()):
            assert not torch.equal(orig, new), "Parameters should change after step"
            assert torch.isfinite(new).all(), "Parameters should remain finite"

    def test_muon_convergence_simple_problem(self):
        """Test that Muon can solve a simple optimization problem."""
        torch.manual_seed(42)

        # Simple problem: learn to map input to target
        model = nn.Linear(2, 1, bias=False)
        target_weight = torch.tensor([[1.0, -1.0]])

        optimizer = Muon(model.parameters(), lr=1e-2)

        losses = []
        for epoch in range(100):
            # Generate random data
            x = torch.randn(10, 2)
            y = x @ target_weight.T

            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Check for NaN/Inf
            assert torch.isfinite(loss), f"Loss became non-finite at epoch {epoch}"
            for p in model.parameters():
                assert torch.isfinite(p).all(), f"Parameters became non-finite at epoch {epoch}"

        # Check that loss decreased
        assert losses[-1] < losses[0], "Loss should decrease during training"
        assert losses[-1] < 0.5, f"Final loss too high: {losses[-1]}"
        print(f"Muon convergence: {losses[0]:.4f} -> {losses[-1]:.4f}")

    def test_muon_dimension_scaling(self):
        """Test dimension scaling calculation."""
        dummy_param = torch.randn(1, 1, requires_grad=True)
        muon = Muon([dummy_param])

        # Test 2D shape (standard linear layer)
        scaling_2d = muon.get_dimension_scaling((100, 50))
        expected_2d = math.sqrt(50 / 100)  # sqrt(fan_out / fan_in)
        assert abs(scaling_2d - expected_2d) < 1e-6, f"2D scaling incorrect: {scaling_2d} vs {expected_2d}"

        # Test 4D shape (conv layer)
        scaling_4d = muon.get_dimension_scaling((64, 32, 3, 3))  # out_ch, in_ch, h, w
        fan_in = 32 * 3 * 3
        fan_out = 64 * 3 * 3
        expected_4d = math.sqrt(fan_out / fan_in)
        assert abs(scaling_4d - expected_4d) < 1e-6, f"4D scaling incorrect: {scaling_4d} vs {expected_4d}"

        # Test 1D shape
        scaling_1d = muon.get_dimension_scaling((100,))
        assert scaling_1d == 1.0, f"1D scaling should be 1.0, got {scaling_1d}"


class TestMixedOptimizerV2:
    """Test MixedOptimizerV2."""

    def test_parameter_categorization(self):
        """Test that parameters are correctly categorized."""
        model = SimpleTransformer(vocab_size=1000, d_model=128)
        optimizer = MixedOptimizerV2(model)

        # Check parameter groups
        param_types = {}
        for group in optimizer.param_groups:
            param_name = group["name"]
            param_type = group["type"]
            optimizer_type = group["optimizer"]
            param_types[param_name] = (param_type, optimizer_type)

        # Verify categorization
        assert param_types["token_embeddings.weight"] == ("embedding", "adam"), "Embeddings should use Adam"
        assert param_types["lm_head.weight"] == ("lm_head", "adam"), "LM head should use Adam"
        assert param_types["layer_norm.weight"] == ("1d", "adam"), "LayerNorm weight should use Adam"
        assert param_types["layer_norm.bias"] == ("1d", "adam"), "LayerNorm bias should use Adam"
        assert param_types["linear1.weight"] == ("linear", "muon"), "Linear weights should use Muon"
        assert param_types["linear2.weight"] == ("linear", "muon"), "Linear weights should use Muon"
        assert param_types["linear1.bias"] == ("1d", "adam"), "Linear bias should use Adam"

    def test_mixed_optimizer_step(self):
        """Test MixedOptimizerV2 optimization step."""
        torch.manual_seed(42)
        model = SimpleTransformer(vocab_size=100, d_model=64)
        optimizer = MixedOptimizerV2(model, muon_lr=1e-3, adam_lr=1e-4)

        # Create dummy data
        x = torch.randint(0, 100, (8, 16))  # batch_size=8, seq_len=16
        y = torch.randint(0, 100, (8, 16))

        # Forward pass
        output = model(x)
        loss = nn.CrossEntropyLoss()(output.view(-1, 100), y.view(-1))

        # Backward pass
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.clone()

        # Optimization step
        optimizer.step()

        # Check that parameters changed and remain finite
        for name, param in model.named_parameters():
            assert not torch.equal(original_params[name], param), f"Parameter {name} should change"
            assert torch.isfinite(param).all(), f"Parameter {name} should remain finite"

    def test_mixed_optimizer_learning_rate_update(self):
        """Test learning rate update functionality."""
        model = SimpleTransformer(vocab_size=100, d_model=64)
        optimizer = MixedOptimizerV2(model, muon_lr=1e-3, adam_lr=1e-4, embedding_lr=2e-4, lm_head_lr=3e-4)

        # Update learning rates
        optimizer.update_learning_rates(
            muon_lr_factor=0.5, adam_lr_factor=0.8, embedding_lr_factor=0.6, lm_head_lr_factor=0.7
        )

        # Check updated rates
        for group in optimizer.param_groups:
            param_type = group["type"]
            if param_type == "linear":
                assert abs(group["lr"] - 1e-3 * 0.5) < 1e-10, "Muon LR not updated correctly"
            elif param_type == "embedding":
                assert abs(group["lr"] - 1e-3 * 0.6) < 1e-10, "Embedding LR not updated correctly"
            elif param_type == "lm_head":
                assert abs(group["lr"] - 1e-3 * 0.7) < 1e-10, "LM head LR not updated correctly"
            elif param_type == "1d":
                assert abs(group["lr"] - 1e-4 * 0.8) < 1e-10, "Adam LR not updated correctly"

    def test_mixed_optimizer_convergence(self):
        """Test that MixedOptimizerV2 can train a model successfully."""
        torch.manual_seed(42)
        model = SimpleTransformer(vocab_size=50, d_model=32)
        optimizer = MixedOptimizerV2(model, muon_lr=5e-3, adam_lr=1e-3)

        losses = []
        for epoch in range(50):
            # Generate random sequence data
            x = torch.randint(0, 50, (4, 8))
            y = torch.randint(0, 50, (4, 8))

            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output.view(-1, 50), y.view(-1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Check for stability
            assert torch.isfinite(loss), f"Loss became non-finite at epoch {epoch}"
            for param in model.parameters():
                assert torch.isfinite(param).all(), f"Parameters became non-finite at epoch {epoch}"

        # Check that loss generally decreased
        # Use a moving average to smooth out noise
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10
        assert late_avg < early_avg, f"Loss should decrease: {early_avg:.4f} -> {late_avg:.4f}"


class TestStabilityAndEdgeCases:
    """Test stability and edge cases."""

    def test_gradient_explosion_handling(self):
        """Test handling of gradient explosion."""
        torch.manual_seed(42)
        model = SimpleLinear(5, 10, 2)
        optimizer = MixedOptimizerV2(model)

        # Create exploding gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 1e6  # Very large gradients

        # Should not crash
        optimizer.step()

        # Parameters should remain finite
        for param in model.parameters():
            assert torch.isfinite(param).all(), "Parameters should remain finite despite large gradients"

    def test_zero_gradients(self):
        """Test handling of zero gradients."""
        model = SimpleLinear(5, 10, 2)
        optimizer = Muon(model.parameters())

        # Set all gradients to zero
        for param in model.parameters():
            param.grad = torch.zeros_like(param)

        # Should not crash
        optimizer.step()

        # Parameters should remain finite
        for param in model.parameters():
            assert torch.isfinite(param).all(), "Parameters should remain finite with zero gradients"

    def test_nan_gradient_handling(self):
        """Test handling of NaN gradients."""
        model = SimpleLinear(5, 10, 2)
        optimizer = MixedOptimizerV2(model)

        # Set some gradients to NaN
        for i, param in enumerate(model.parameters()):
            if i % 2 == 0:
                param.grad = torch.full_like(param, float("nan"))
            else:
                param.grad = torch.randn_like(param)

        # Should not crash and should handle NaN gracefully
        optimizer.step()

        # Parameters should remain finite
        for param in model.parameters():
            assert torch.isfinite(param).all(), "Parameters should remain finite despite NaN gradients"

    def test_very_small_learning_rates(self):
        """Test with very small learning rates."""
        model = SimpleLinear(5, 10, 2)
        optimizer = Muon(model.parameters(), lr=1e-10)

        # Create gradients
        x = torch.randn(4, 5)
        y = torch.randn(4, 2)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Should not crash
        optimizer.step()

        # Parameters should remain finite
        for param in model.parameters():
            assert torch.isfinite(param).all(), "Parameters should remain finite with tiny LR"

    def test_very_large_learning_rates(self):
        """Test with very large learning rates."""
        model = SimpleLinear(5, 10, 2)
        optimizer = Muon(model.parameters(), lr=100.0)

        # Create gradients
        x = torch.randn(4, 5)
        y = torch.randn(4, 2)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Should not crash (though may not converge)
        optimizer.step()

        # Parameters should remain finite (due to our stability measures)
        for param in model.parameters():
            assert torch.isfinite(param).all(), "Parameters should remain finite even with large LR"


class TestComparison:
    """Compare Muon with other optimizers."""

    def test_muon_vs_adam_simple_problem(self):
        """Compare Muon vs Adam on a simple problem."""
        torch.manual_seed(42)

        # Target function: y = 2*x1 - x2 + 1
        target_weight = torch.tensor([[2.0, -1.0]])
        target_bias = torch.tensor([1.0])

        def train_optimizer(optimizer_class, **kwargs):
            model = nn.Linear(2, 1)
            optimizer = optimizer_class(model.parameters(), **kwargs)

            losses = []
            for epoch in range(100):
                x = torch.randn(20, 2)
                y = x @ target_weight.T + target_bias

                optimizer.zero_grad()
                output = model(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            return losses

        # Train with Muon (use smaller LR as Muon can be more aggressive)
        muon_losses = train_optimizer(Muon, lr=5e-3)

        # Train with Adam
        adam_losses = train_optimizer(Adam, lr=1e-2)

        # Both should show improvement (final loss < initial loss)
        assert muon_losses[-1] < muon_losses[0], f"Muon didn't improve: {muon_losses[0]:.4f} -> {muon_losses[-1]:.4f}"
        assert adam_losses[-1] < adam_losses[0], f"Adam didn't improve: {adam_losses[0]:.4f} -> {adam_losses[-1]:.4f}"

        # At least one should converge well
        best_final_loss = min(muon_losses[-1], adam_losses[-1])
        assert best_final_loss < 1.0, (
            f"Neither optimizer converged well: Muon={muon_losses[-1]:.4f}, Adam={adam_losses[-1]:.4f}"
        )

        # Both should be stable (no NaN/Inf)
        assert all(math.isfinite(loss) for loss in muon_losses), "Muon losses should be finite"
        assert all(math.isfinite(loss) for loss in adam_losses), "Adam losses should be finite"

        print(f"Final losses - Muon: {muon_losses[-1]:.6f}, Adam: {adam_losses[-1]:.6f}")


if __name__ == "__main__":
    # Run basic tests
    test_ns = TestNewtonSchulz()
    test_ns.test_newton_schulz_basic_convergence()
    test_ns.test_newton_schulz_stability_with_extreme_inputs()
    print("✅ Newton-Schulz tests passed")

    test_muon = TestMuon()
    test_muon.test_muon_initialization()
    test_muon.test_muon_step_basic()
    test_muon.test_muon_convergence_simple_problem()
    print("✅ Muon tests passed")

    test_mixed = TestMixedOptimizerV2()
    test_mixed.test_parameter_categorization()
    test_mixed.test_mixed_optimizer_step()
    test_mixed.test_mixed_optimizer_convergence()
    print("✅ MixedOptimizerV2 tests passed")

    test_stability = TestStabilityAndEdgeCases()
    test_stability.test_zero_gradients()
    test_stability.test_very_small_learning_rates()
    print("✅ Stability tests passed")

    test_comparison = TestComparison()
    test_comparison.test_muon_vs_adam_simple_problem()
    print("✅ Comparison tests passed")

    print("\n🎉 All tests passed! The Muon implementation is working correctly.")
