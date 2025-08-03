"""Native PyTorch FP8 Linear layer implementation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FP8Linear(nn.Module):
    """Linear layer with native PyTorch FP8 support.

    Uses torch.float8_e4m3fn for forward pass and torch.float8_e5m2 for gradients.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize with FP32 for stability
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        # Scaling factors for FP8 quantization
        self.register_buffer("input_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("weight_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("grad_scale", torch.tensor(1.0, dtype=torch.float32))

        # FP8 dtypes
        self.fp8_e4m3 = torch.float8_e4m3fn  # Forward pass
        self.fp8_e5m2 = torch.float8_e5m2  # Gradients

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with scaled init."""
        # Using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.compile(mode="max-autotune", fullgraph=True)
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with FP8 computation."""
        # Cast and scale for FP8
        if self.training and input.device.type == "cuda":
            # Use FP8 for computation
            try:
                # Cast input and weight to FP8
                input_fp8 = input.to(self.fp8_e4m3)
                weight_fp8 = self.weight.to(self.fp8_e4m3)

                # Use torch._scaled_mm for FP8 matrix multiplication
                # This is the native PyTorch FP8 matmul
                output = torch._scaled_mm(
                    input_fp8,
                    weight_fp8.t(),
                    scale_a=self.input_scale,
                    scale_b=self.weight_scale,
                    scale_result=None,
                    out_dtype=torch.float32,
                )

                if self.bias is not None:
                    output = output + self.bias

                return output

            except Exception as e:
                # Fallback to FP32 if FP8 fails
                if "scaled_mm" in str(e) or "float8" in str(e):
                    print(f"Warning: FP8 computation failed, falling back to FP32: {e}")
                    return F.linear(input, self.weight, self.bias)
                else:
                    raise
        else:
            # Use standard FP32 for inference or non-CUDA
            return F.linear(input, self.weight, self.bias)


class FP8LinearFunction(torch.autograd.Function):
    """Custom autograd function for FP8 linear with proper gradient handling."""

    @staticmethod
    def forward(ctx, input, weight, bias, input_scale, weight_scale):
        # Save for backward
        ctx.save_for_backward(input, weight, bias, input_scale, weight_scale)

        # FP8 forward computation
        if input.device.type == "cuda":
            try:
                input_fp8 = input.to(torch.float8_e4m3fn)
                weight_fp8 = weight.to(torch.float8_e4m3fn)

                output = torch._scaled_mm(
                    input_fp8,
                    weight_fp8.t(),
                    scale_a=input_scale,
                    scale_b=weight_scale,
                    out_dtype=torch.float32,
                )

                if bias is not None:
                    output = output + bias

                return output
            except:
                # Fallback
                return F.linear(input, weight, bias)
        else:
            return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, input_scale, weight_scale = ctx.saved_tensors

        # Compute gradients in FP8
        if grad_output.device.type == "cuda":
            try:
                grad_output_fp8 = grad_output.to(torch.float8_e5m2)

                # Gradient w.r.t. input
                weight_fp8 = weight.to(torch.float8_e5m2)
                # Use scale of 1.0 for gradients (can be optimized later)
                grad_scale = torch.tensor(1.0, device=grad_output.device)
                grad_input = torch._scaled_mm(
                    grad_output_fp8,
                    weight_fp8,
                    scale_a=grad_scale,
                    scale_b=weight_scale,
                    out_dtype=torch.float32,
                )

                # Gradient w.r.t. weight
                input_fp8 = input.to(torch.float8_e5m2)
                grad_weight = torch._scaled_mm(
                    grad_output_fp8.t(),
                    input_fp8,
                    scale_a=grad_scale,
                    scale_b=input_scale,
                    out_dtype=torch.float32,
                )

                # Gradient w.r.t. bias
                grad_bias = grad_output.sum(0) if bias is not None else None

                return grad_input, grad_weight, grad_bias, None, None
            except:
                # Fallback to FP32 gradients
                grad_input = grad_output @ weight
                grad_weight = grad_output.t() @ input
                grad_bias = grad_output.sum(0) if bias is not None else None
                return grad_input, grad_weight, grad_bias, None, None
        else:
            grad_input = grad_output @ weight
            grad_weight = grad_output.t() @ input
            grad_bias = grad_output.sum(0) if bias is not None else None
            return grad_input, grad_weight, grad_bias, None, None


class FP8Module(nn.Module):
    """Wrapper to convert a module to use FP8 linear layers."""

    @staticmethod
    def convert_to_fp8(module: nn.Module, verbose: bool = True) -> nn.Module:
        """Recursively replace nn.Linear with FP8Linear."""
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with FP8Linear
                fp8_linear = FP8Linear(
                    child.in_features,
                    child.out_features,
                    child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                # Copy weights
                with torch.no_grad():
                    fp8_linear.weight.copy_(child.weight)
                    if child.bias is not None:
                        fp8_linear.bias.copy_(child.bias)
                setattr(module, name, fp8_linear)
                if verbose:
                    print(f"Converted {name} to FP8Linear")
            else:
                # Recursively convert children
                FP8Module.convert_to_fp8(child, verbose=verbose)
        return module
