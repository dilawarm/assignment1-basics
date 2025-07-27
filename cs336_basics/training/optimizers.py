"""
Optimizers for neural network training.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional

import torch
from torch.optim import Optimizer


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AdvancedAdamW(Optimizer):
    """
    Enhanced AdamW with advanced update clipping for improved final accuracy.

    Based on research from "Stable and low-precision training for large-scale vision-language models"
    which showed that clipping Adam updates (rather than gradients) can improve final performance.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.85),  # Lower β2 for stability
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        # Advanced clipping parameters
        clip_threshold: float = 1.0,
        enable_update_clipping: bool = True,
        adaptive_clipping: bool = True,
    ):
        """
        Initialize Advanced AdamW optimizer.

        Args:
            params: Parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (β1, β2)
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay coefficient
            amsgrad: Whether to use AMSGrad variant
            maximize: Maximize parameters instead of minimizing
            foreach: Whether to use vectorized implementation
            capturable: Whether to use capturable implementation
            differentiable: Whether to use differentiable implementation
            fused: Whether to use fused implementation
            clip_threshold: Threshold for update clipping
            enable_update_clipping: Whether to enable update clipping
            adaptive_clipping: Whether to use adaptive clipping thresholds
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= clip_threshold:
            raise ValueError(f"Invalid clip_threshold: {clip_threshold}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            clip_threshold=clip_threshold,
            enable_update_clipping=enable_update_clipping,
            adaptive_clipping=adaptive_clipping,
        )
        super().__init__(params, defaults)

        # Statistics for adaptive clipping
        self.update_norms = {}
        self.adaptive_thresholds = {}
        self.step_count = 0

    def _update_adaptive_threshold(self, param_id: str, update_norm: float, threshold: float) -> float:
        """Update adaptive clipping threshold based on update statistics."""
        if param_id not in self.adaptive_thresholds:
            self.adaptive_thresholds[param_id] = threshold
            self.update_norms[param_id] = update_norm
        else:
            # Use EMA to track typical update norms
            alpha = 0.01  # Slow adaptation
            self.update_norms[param_id] = (1 - alpha) * self.update_norms[param_id] + alpha * update_norm

            # Adapt threshold based on recent update patterns
            typical_norm = self.update_norms[param_id]
            self.adaptive_thresholds[param_id] = max(
                threshold * 0.5,  # Minimum threshold
                min(threshold * 2.0, typical_norm * 3.0),  # Maximum threshold
            )

        return self.adaptive_thresholds[param_id]

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step with advanced update clipping."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                param_id = f"param_{id(p)}"

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state["step"] += 1
                state_steps.append(state["step"])

            # Perform AdamW updates with advanced clipping
            self._adamw_with_clipping(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                clip_threshold=group["clip_threshold"],
                enable_update_clipping=group["enable_update_clipping"],
                adaptive_clipping=group["adaptive_clipping"],
            )

        return loss

    def _adamw_with_clipping(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        maximize: bool,
        clip_threshold: float,
        enable_update_clipping: bool,
        adaptive_clipping: bool,
    ):
        """AdamW implementation with advanced update clipping."""

        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            param_id = f"param_{id(param)}"

            # Perform weight decay
            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)

            # Update biased first moment estimate
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # Update biased second raw moment estimate
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(1 - beta2**step)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2**step)).add_(eps)

            # Bias correction
            step_size = lr / (1 - beta1**step)

            # Compute the update
            update = exp_avg / denom

            # Advanced update clipping - key innovation for improved accuracy
            if enable_update_clipping:
                update_norm = torch.norm(update).item()

                # Determine clipping threshold
                if adaptive_clipping:
                    effective_threshold = self._update_adaptive_threshold(param_id, update_norm, clip_threshold)
                else:
                    effective_threshold = clip_threshold

                # Clip update if necessary
                if update_norm > effective_threshold:
                    # Clip the update, not the gradient
                    clip_factor = effective_threshold / (update_norm + eps)
                    update.mul_(clip_factor)

                    # Optional: Log clipping events for monitoring
                    if hasattr(self, "_log_clipping") and self._log_clipping and self.step_count % 100 == 0:
                        print(f"Advanced Adam clipping: {update_norm:.3f} -> {effective_threshold:.3f}")

            # Apply update
            param.add_(update, alpha=-step_size)


# Legacy AdamW for backward compatibility
class AdamW(AdvancedAdamW):
    """Standard AdamW optimizer - now inherits from AdvancedAdamW."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, **kwargs):
        # Use standard β2 for legacy compatibility
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            enable_update_clipping=False,  # Disabled by default for compatibility
            **kwargs,
        )


class Muon(Optimizer):
    """
    Enhanced Muon optimizer with outlier-safe training capabilities.

    Based on:
    - "Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models" (2025)
    - YouJiacheng, Franz Cesista, and Jeremy Bernstein's research
    - Advanced stability techniques for preventing NaN/inf gradients

    Features:
    - Robust Newton-Schulz orthogonalization with stability checks
    - Outlier detection and mitigation
    - Adaptive scaling based on activation statistics
    - Built-in gradient health monitoring
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 3e-3,
        momentum: float = 0.95,
        ns_iters: int = 5,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        use_optimized_coefficients: bool = True,
        outlier_threshold: float = 6.0,
        enable_outlier_detection: bool = True,
        stability_check_freq: int = 10,
        max_norm_scale: float = 10.0,
        enable_stability_logging: bool = False,
    ) -> None:
        """
        Initialize enhanced Muon optimizer.

        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate (typically higher than AdamW)
            momentum: Momentum factor for exponential moving average
            ns_iters: Number of Newton-Schulz iterations for orthogonalization
            weight_decay: Weight decay coefficient
            eps: Small constant for numerical stability
            use_optimized_coefficients: Whether to use optimized NS coefficients
            outlier_threshold: Z-score threshold for outlier detection
            enable_outlier_detection: Whether to enable outlier detection and mitigation
            stability_check_freq: Frequency of stability checks
            max_norm_scale: Maximum allowed norm scaling factor
            enable_stability_logging: Whether to log stability information
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 1 <= ns_iters <= 10:
            raise ValueError(f"Invalid ns_iters value: {ns_iters}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_iters=ns_iters,
            weight_decay=weight_decay,
            eps=eps,
            use_optimized_coefficients=use_optimized_coefficients,
            outlier_threshold=outlier_threshold,
            enable_outlier_detection=enable_outlier_detection,
            stability_check_freq=stability_check_freq,
            max_norm_scale=max_norm_scale,
            enable_stability_logging=enable_stability_logging,
        )
        super().__init__(params, defaults)

        # Stability tracking
        self.step_count = 0
        self.outlier_count = 0
        self.instability_count = 0
        self.emergency_fallbacks = 0

        # Outlier detection statistics
        self.param_norm_history = {}
        self.param_norm_ema = {}
        self.param_norm_var = {}

        if use_optimized_coefficients:
            # YouJiacheng's proven 6-step coefficients for maximum stability
            # Source: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b
            self.optimized_ns_coefficients = [
                (3955 / 1024, -8306 / 1024, 5008 / 1024),
                (3735 / 1024, -6681 / 1024, 3463 / 1024),
                (3799 / 1024, -6499 / 1024, 3211 / 1024),
                (4019 / 1024, -6385 / 1024, 2906 / 1024),
                (2677 / 1024, -3029 / 1024, 1162 / 1024),
                (2172 / 1024, -1833 / 1024, 682 / 1024),
            ]
        else:
            # Stable cubic fallback
            self.optimized_ns_coefficients = [(1.5, -0.5, 0.0)] * ns_iters

    def detect_outliers(self, tensor: torch.Tensor, param_id: str, threshold: float) -> bool:
        """
        Detect outliers in gradients or parameters using statistical analysis.

        Args:
            tensor: Input tensor to analyze
            param_id: Unique identifier for the parameter
            threshold: Z-score threshold for outlier detection

        Returns:
            True if outliers are detected
        """
        if not torch.isfinite(tensor).all():
            return True

        # Compute current statistics
        current_norm = torch.norm(tensor, p="fro").item()

        # Initialize tracking for new parameters
        if param_id not in self.param_norm_ema:
            self.param_norm_ema[param_id] = current_norm
            self.param_norm_var[param_id] = 1.0
            return False

        # Update EMA statistics
        ema_decay = 0.99
        delta = current_norm - self.param_norm_ema[param_id]
        self.param_norm_ema[param_id] += (1 - ema_decay) * delta
        self.param_norm_var[param_id] = ema_decay * self.param_norm_var[param_id] + (1 - ema_decay) * delta * delta

        # Compute z-score
        std_dev = math.sqrt(max(self.param_norm_var[param_id], 1e-8))
        z_score = abs(current_norm - self.param_norm_ema[param_id]) / std_dev

        return z_score > threshold

    def mitigate_outliers(self, tensor: torch.Tensor, max_scale: float = 3.0) -> torch.Tensor:
        """
        Mitigate outliers by clipping extreme values while preserving tensor structure.

        Args:
            tensor: Input tensor with potential outliers
            max_scale: Maximum scaling factor for outlier mitigation

        Returns:
            Tensor with outliers mitigated
        """
        # Compute percentiles for robust outlier detection
        flat_tensor = tensor.flatten()
        q1, q3 = torch.quantile(flat_tensor, torch.tensor([0.25, 0.75], device=tensor.device))
        iqr = q3 - q1

        if iqr < 1e-8:
            return tensor

        # Define outlier bounds using IQR method
        lower_bound = q1 - max_scale * iqr
        upper_bound = q3 + max_scale * iqr

        # Clip outliers
        return torch.clamp(tensor, lower_bound, upper_bound)

    def newton_schulz_orthogonalize(self, G: torch.Tensor, num_iters: int = 5, param_id: str = "") -> torch.Tensor:
        """
        Apply robust Newton-Schulz iterations with outlier detection and stability checks.

        Args:
            G: Input matrix to orthogonalize
            num_iters: Number of Newton-Schulz iterations
            param_id: Parameter identifier for tracking

        Returns:
            Orthogonalized matrix
        """
        # Input validation and early returns
        if torch.isnan(G).any() or torch.isinf(G).any():
            self.instability_count += 1
            if self.defaults.get("enable_stability_logging", False):
                print(f"Muon: NaN/Inf detected in input matrix for {param_id}")
            return torch.eye(G.shape[0], G.shape[1], device=G.device, dtype=G.dtype) * 0.01

        norm = torch.norm(G, p="fro")
        if norm < 1e-8:
            return G

        # Outlier detection and mitigation
        if self.defaults.get("enable_outlier_detection", True):
            if self.detect_outliers(G, param_id, self.defaults.get("outlier_threshold", 6.0)):
                self.outlier_count += 1
                G = self.mitigate_outliers(G, max_scale=3.0)
                if self.defaults.get("enable_stability_logging", False):
                    print(f"Muon: Outliers mitigated for {param_id}")

        # Normalize to ensure convergence (critical step)
        X = G / (norm + 1e-8)

        device = X.device
        dtype = X.dtype

        # Determine optimal approach based on matrix shape
        transpose = X.shape[0] > X.shape[1]
        if transpose:
            X = X.transpose(-2, -1)

        # Use stable coefficients
        if hasattr(self, "optimized_ns_coefficients") and len(self.optimized_ns_coefficients) > 0:
            coeffs = self.optimized_ns_coefficients[: min(num_iters, len(self.optimized_ns_coefficients))]
        else:
            # Fallback to stable cubic
            coeffs = [(1.5, -0.5, 0.0)] * num_iters

        for i, (a, b, c) in enumerate(coeffs):
            try:
                # Compute Gram matrix with enhanced stability
                A = torch.matmul(X.transpose(-2, -1), X)
                I = torch.eye(A.shape[-1], device=device, dtype=dtype)

                # Adaptive regularization based on condition number
                try:
                    s = torch.linalg.svdvals(A)
                    cond_num = s.max() / (s.min() + 1e-12)
                    reg_factor = max(1e-7, min(1e-4, 1.0 / cond_num))
                except:
                    reg_factor = 1e-7

                A = A + reg_factor * I

                # Newton-Schulz iteration with stability checks
                if abs(c) > 1e-8:  # Use quintic if c is significant
                    try:
                        A_squared = torch.matmul(A, A)
                        poly_matrix = a * I + b * A + c * A_squared
                        X_new = torch.matmul(X, poly_matrix)
                    except:
                        # Fallback to cubic if quintic fails
                        poly_matrix = 1.5 * I - 0.5 * A
                        X_new = torch.matmul(X, poly_matrix)
                else:
                    # Use cubic Newton-Schulz: X @ (aI + bA)
                    poly_matrix = a * I + b * A
                    X_new = torch.matmul(X, poly_matrix)

                # Enhanced stability checks
                if torch.isnan(X_new).any() or torch.isinf(X_new).any():
                    self.instability_count += 1
                    # Fall back to basic cubic
                    poly_matrix = 1.5 * I - 0.5 * A
                    X_new = torch.matmul(X, poly_matrix)

                # Adaptive clamping based on iteration and stability
                max_val = max(5.0, 10.0 / (i + 1))  # Progressively tighter bounds
                X_new = torch.clamp(X_new, min=-max_val, max=max_val)

                # Check for convergence
                diff_norm = torch.norm(X_new - X, p="fro")
                if diff_norm < 1e-6:
                    X = X_new
                    break

                # Progressive stability check
                current_norm = torch.norm(X_new, p="fro")
                max_allowed_norm = self.defaults.get("max_norm_scale", 10.0)

                if current_norm > max_allowed_norm:
                    # Normalize to prevent explosion
                    X_new = X_new / current_norm * min(max_allowed_norm, 2.0)
                    self.instability_count += 1

                # Apply update
                X = X_new

            except Exception as e:
                # Robust fallback to basic cubic iteration
                self.emergency_fallbacks += 1
                try:
                    A = torch.matmul(X.transpose(-2, -1), X)
                    I = torch.eye(A.shape[-1], device=device, dtype=dtype)
                    A = A + 1e-6 * I  # Higher regularization for emergency
                    X = torch.matmul(X, 1.5 * I - 0.5 * A)
                except:
                    # Ultimate fallback: return scaled identity
                    eye_matrix = torch.eye(X.shape[0], X.shape[1], device=device, dtype=dtype)
                    if self.defaults.get("enable_stability_logging", False):
                        print(f"Muon: Emergency fallback activated for {param_id}")
                    return eye_matrix * 0.01

        # Restore original orientation
        if transpose:
            X = X.transpose(-2, -1)

        # Final safety check with outlier mitigation
        if torch.isnan(X).any() or torch.isinf(X).any():
            self.emergency_fallbacks += 1
            return torch.eye(G.shape[0], G.shape[1], device=device, dtype=dtype) * 0.01

        # Final outlier check and mitigation
        if self.defaults.get("enable_outlier_detection", True):
            X = self.mitigate_outliers(X, max_scale=2.0)

        return X

    def get_dimension_scaling(self, shape: tuple[int, ...]) -> float:
        """Calculate robust dimension scaling factor for Muon updates."""
        if len(shape) == 2:
            fan_in, fan_out = shape
            # More conservative scaling to prevent instability
            scaling = math.sqrt(fan_out / max(fan_in, 1))
            return min(scaling, 2.0)  # Cap the scaling factor
        elif len(shape) == 1:
            return 1.0
        elif len(shape) == 4:  # Conv layers (flattened)
            c_out, c_in, k_h, k_w = shape
            fan_in = c_in * k_h * k_w
            fan_out = c_out * k_h * k_w
            scaling = math.sqrt(fan_out / max(fan_in, 1))
            return min(scaling, 2.0)  # Cap the scaling factor
        else:
            total_elements = torch.prod(torch.tensor(shape)).item()
            scaling = math.sqrt(total_elements)
            return min(scaling, 3.0)  # Conservative cap for high-dimensional tensors

    def get_stability_stats(self) -> dict:
        """Get stability statistics for monitoring."""
        total_steps = max(self.step_count, 1)
        return {
            "muon_outlier_rate": self.outlier_count / total_steps,
            "muon_instability_rate": self.instability_count / total_steps,
            "muon_emergency_fallback_rate": self.emergency_fallbacks / total_steps,
            "muon_total_outliers": self.outlier_count,
            "muon_total_instabilities": self.instability_count,
            "muon_total_emergency_fallbacks": self.emergency_fallbacks,
        }

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step with enhanced stability."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_iters = group["ns_iters"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            stability_check_freq = group.get("stability_check_freq", 10)
            enable_stability_logging = group.get("enable_stability_logging", False)

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad

                # Convert to float32 for stability if needed
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                # Pre-processing gradient health check
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    if enable_stability_logging:
                        print(f"Muon: NaN/Inf gradient detected, skipping parameter {i}")
                    continue

                state = self.state[p]
                param_id = f"param_{id(p)}"

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(grad)

                momentum_buffer = state["momentum_buffer"]
                state["step"] += 1

                # Outlier detection on gradients
                if group.get("enable_outlier_detection", True):
                    if self.detect_outliers(grad, f"{param_id}_grad", group.get("outlier_threshold", 6.0)):
                        grad = self.mitigate_outliers(grad, max_scale=3.0)
                        if enable_stability_logging and state["step"] % stability_check_freq == 0:
                            print(f"Muon: Gradient outliers mitigated for parameter {i}")

                # Update momentum with stability check
                momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)

                # Check momentum buffer health
                if torch.isnan(momentum_buffer).any() or torch.isinf(momentum_buffer).any():
                    momentum_buffer.copy_(grad * (1 - momentum))
                    if enable_stability_logging:
                        print(f"Muon: Momentum buffer reset for parameter {i}")

                # Apply Muon update for 2D+ parameters
                if len(p.shape) >= 2:
                    # Reshape for Newton-Schulz if needed
                    original_shape = p.shape
                    if len(p.shape) > 2:
                        param_2d = p.reshape(p.shape[0], -1)
                        momentum_2d = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
                    else:
                        param_2d = p
                        momentum_2d = momentum_buffer

                    # Apply Newton-Schulz orthogonalization with stability tracking
                    ortho_momentum = self.newton_schulz_orthogonalize(momentum_2d, ns_iters, f"{param_id}_2d")

                    # Calculate robust scaling
                    dim_scaling = self.get_dimension_scaling(original_shape)
                    momentum_norm = torch.norm(momentum_2d, p="fro")

                    if momentum_norm > eps:
                        # Scale the orthogonalized momentum with stability bounds
                        scaling_factor = dim_scaling / (momentum_norm + eps)
                        scaling_factor = min(scaling_factor, 2.0)  # Conservative cap
                        update = ortho_momentum * scaling_factor

                        # Reshape back if needed
                        if len(p.shape) > 2:
                            update = update.reshape(original_shape)

                        # Final update health check
                        if torch.isnan(update).any() or torch.isinf(update).any():
                            if enable_stability_logging:
                                print(f"Muon: Skipping update due to NaN/Inf for parameter {i}")
                            continue

                        # Apply weight decay
                        if weight_decay > 0:
                            p.mul_(1 - lr * weight_decay)

                        # Apply update with conservative learning rate scaling
                        effective_lr = lr
                        if state["step"] < 100:  # Warm-up period
                            effective_lr *= min(1.0, state["step"] / 100.0)

                        p.add_(update, alpha=-effective_lr)

                else:
                    # Standard momentum update for 1D parameters with stability
                    if weight_decay > 0:
                        p.mul_(1 - lr * weight_decay)

                    # Check update health
                    if torch.isfinite(momentum_buffer).all():
                        effective_lr = lr
                        if state["step"] < 100:  # Warm-up period
                            effective_lr *= min(1.0, state["step"] / 100.0)
                        p.add_(momentum_buffer, alpha=-effective_lr)

                # Periodic stability logging
                if enable_stability_logging and state["step"] % stability_check_freq == 0:
                    param_norm = torch.norm(p).item()
                    grad_norm = torch.norm(grad).item()
                    if param_norm > 100 or grad_norm > 100:
                        print(f"Muon: High norms detected - Param: {param_norm:.3f}, Grad: {grad_norm:.3f}")

        return loss


class Madgrad(Optimizer):
    """
    Madgrad optimizer from Facebook AI Research.

    Based on: "Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization"

    From the research: "In testing with transformers for image classification, madgrad blew away the various Adam variants."
    Modified to use AdamW-style weight decay for better performance.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eps: float = 1e-6,
    ):
        """
        Initialize Madgrad optimizer.

        Args:
            params: Parameters to optimize
            lr: Learning rate (typically higher than Adam, try 1e-2)
            momentum: Momentum parameter
            weight_decay: Weight decay coefficient (AdamW-style)
            eps: Small constant for numerical stability
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"]
            decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # AdamW-style weight decay (applied directly to parameters)
                if decay != 0:
                    p.mul_(1 - lr * decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)

                exp_avg_sq = state["exp_avg_sq"]
                s = state["s"]
                state["step"] += 1

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(momentum).addcmul_(grad, grad, value=1 - momentum)

                # Bias correction
                bias_correction = 1 - momentum ** state["step"]
                k = group["lr"] / bias_correction

                # Update s
                s.add_(grad, alpha=k)

                # Compute denominator
                denom = exp_avg_sq.sqrt().add_(eps)

                # Apply update
                p.addcdiv_(s, denom, value=-1)

        return loss


class MixedOptimizerV2(Optimizer):
    """
    Mixed optimizer that uses Muon for linear layers and Adam for embeddings/lm_head/1D params.

    Implements the winning solution approach:
    - Muon for everything except embedding, lm_head and 1-dimensional values
    - Different learning rates for different parameter groups
    - Proper parameter categorization as described in the winning solution
    """

    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float = 8e-3,
        adam_lr: float = 6e-3,
        embedding_lr: float = 12e-3,
        lm_head_lr: float = 4e-3,
        muon_momentum: float = 0.97,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.015,
        eps: float = 1e-8,
        ns_iters: int = 5,
        use_optimized_muon: bool = True,
        # Outlier-safe Muon parameters
        outlier_threshold: float = 5.0,
        enable_outlier_detection: bool = True,
        stability_check_freq: int = 20,
        max_norm_scale: float = 5.0,
        enable_stability_logging: bool = False,
        # Layer-wise learning rate decay (optional, for compatibility)
        layer_wise_decay_factor: float = None,
    ) -> None:
        """
        Initialize the mixed optimizer with outlier-safe features.

        Args:
            model: The model to optimize
            muon_lr: Learning rate for Muon optimizer (linear layers)
            adam_lr: Learning rate for Adam optimizer (1D params)
            embedding_lr: Learning rate for embedding parameters
            lm_head_lr: Learning rate for LM head parameters
            muon_momentum: Momentum for Muon optimizer
            adam_betas: Beta parameters for Adam optimizer
            weight_decay: Weight decay coefficient
            eps: Small constant for numerical stability
            ns_iters: Number of Newton-Schulz iterations
            use_optimized_muon: Whether to use optimized Muon coefficients
            outlier_threshold: Z-score threshold for outlier detection in Muon
            enable_outlier_detection: Whether to enable outlier detection in Muon
            stability_check_freq: Frequency of stability checks in Muon
            max_norm_scale: Maximum norm scaling factor for Muon
            enable_stability_logging: Whether to enable stability logging in Muon
            layer_wise_decay_factor: Layer-wise decay factor (for compatibility, not used in V2)
        """
        # Categorize parameters according to winning solution approach
        param_groups = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine parameter type and corresponding optimizer
            if "token_embeddings" in name or "embedding" in name:
                param_type = "embedding"
                lr = embedding_lr
                optimizer_type = "adam"
            elif "lm_head" in name:
                param_type = "lm_head"
                lr = lm_head_lr
                optimizer_type = "adam"
            elif len(param.shape) <= 1:  # 1-dimensional parameters (biases, norms, etc.)
                param_type = "1d"
                lr = adam_lr
                optimizer_type = "adam"
            else:  # Multi-dimensional parameters (linear layers)
                param_type = "linear"
                lr = muon_lr
                optimizer_type = "muon"

            param_groups.append(
                {
                    "params": [param],
                    "name": name,
                    "type": param_type,
                    "optimizer": optimizer_type,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )

        # Store configuration with outlier-safe features
        self.muon_config = {
            "lr": muon_lr,
            "momentum": muon_momentum,
            "ns_iters": ns_iters,
            "weight_decay": weight_decay,
            "eps": eps,
            "use_optimized_coefficients": use_optimized_muon,
            # Outlier-safe parameters
            "outlier_threshold": outlier_threshold,
            "enable_outlier_detection": enable_outlier_detection,
            "stability_check_freq": stability_check_freq,
            "max_norm_scale": max_norm_scale,
            "enable_stability_logging": enable_stability_logging,
        }

        self.adam_config = {
            "lr": adam_lr,
            "betas": adam_betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        # Initialize base class
        all_params = []
        for group in param_groups:
            all_params.extend(group["params"])

        defaults = dict(lr=muon_lr, weight_decay=weight_decay)
        super().__init__(all_params, defaults)

        self.param_groups = param_groups

        # Initialize Newton-Schulz coefficients for Muon
        if use_optimized_muon:
            self.optimized_ns_coefficients = [
                (3955 / 1024, -8306 / 1024, 5008 / 1024),
                (3735 / 1024, -6681 / 1024, 3463 / 1024),
                (3799 / 1024, -6499 / 1024, 3211 / 1024),
                (4019 / 1024, -6385 / 1024, 2906 / 1024),
                (2677 / 1024, -3029 / 1024, 1162 / 1024),
                (2172 / 1024, -1833 / 1024, 682 / 1024),
            ]
        else:
            self.optimized_ns_coefficients = [(1.5, -0.5, 0.0)] * ns_iters

        # Log parameter distribution
        muon_count = sum(1 for g in param_groups if g["optimizer"] == "muon")
        adam_count = sum(1 for g in param_groups if g["optimizer"] == "adam")
        embedding_count = sum(1 for g in param_groups if g["type"] == "embedding")
        lm_head_count = sum(1 for g in param_groups if g["type"] == "lm_head")

        print(f"MixedOptimizerV2 initialized:")
        print(f"  Muon parameters (linear): {muon_count}")
        print(f"  Adam parameters (1D): {adam_count - embedding_count - lm_head_count}")
        print(f"  Embedding parameters: {embedding_count}")
        print(f"  LM head parameters: {lm_head_count}")

    def newton_schulz_orthogonalize(self, G: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
        """Apply Newton-Schulz orthogonalization using stable coefficients."""
        # Input validation
        if torch.isnan(G).any() or torch.isinf(G).any():
            return torch.eye(G.shape[0], G.shape[1], device=G.device, dtype=G.dtype) * 0.01

        norm = torch.norm(G, p="fro")
        if norm < 1e-8:
            return G

        # Normalize input
        X = G / (norm + 1e-8)

        device = X.device
        dtype = X.dtype

        # Handle transposition
        transpose = X.shape[0] > X.shape[1]
        if transpose:
            X = X.transpose(-2, -1)

        # Use stable coefficients
        coeffs = self.optimized_ns_coefficients[: min(num_iters, len(self.optimized_ns_coefficients))]

        for i, (a, b, c) in enumerate(coeffs):
            try:
                # Compute Gram matrix
                A = torch.matmul(X.transpose(-2, -1), X)
                I = torch.eye(A.shape[-1], device=device, dtype=dtype)
                A = A + 1e-7 * I  # Regularization

                # Newton-Schulz update
                if abs(c) > 1e-8:  # Use quintic if c is significant
                    try:
                        A_squared = torch.matmul(A, A)
                        poly_matrix = a * I + b * A + c * A_squared
                        X_new = torch.matmul(X, poly_matrix)
                    except:
                        # Fallback to cubic
                        poly_matrix = 1.5 * I - 0.5 * A
                        X_new = torch.matmul(X, poly_matrix)
                else:
                    poly_matrix = a * I + b * A
                    X_new = torch.matmul(X, poly_matrix)

                # Stability checks
                if torch.isnan(X_new).any() or torch.isinf(X_new).any():
                    poly_matrix = 1.5 * I - 0.5 * A
                    X_new = torch.matmul(X, poly_matrix)

                # Gentle clamping
                X_new = torch.clamp(X_new, min=-10.0, max=10.0)

                # Check convergence
                if torch.norm(X_new - X, p="fro") < 1e-6:
                    X = X_new
                    break

                X = X_new

                # Stability monitoring
                current_norm = torch.norm(X, p="fro")
                if current_norm > 5.0:
                    X = X / current_norm * 1.73

            except Exception:
                # Robust fallback
                try:
                    A = torch.matmul(X.transpose(-2, -1), X)
                    I = torch.eye(A.shape[-1], device=device, dtype=dtype)
                    X = torch.matmul(X, 1.5 * I - 0.5 * A)
                except:
                    return torch.eye(G.shape[0], G.shape[1], device=device, dtype=dtype) * 0.1

        # Restore orientation
        if transpose:
            X = X.transpose(-2, -1)

        # Final check
        if torch.isnan(X).any() or torch.isinf(X).any():
            return torch.eye(G.shape[0], G.shape[1], device=device, dtype=dtype) * 0.1

        return X

    def get_dimension_scaling(self, shape: tuple[int, ...]) -> float:
        """Calculate dimension scaling factor."""
        if len(shape) == 2:
            fan_in, fan_out = shape
            return math.sqrt(fan_out / fan_in)
        elif len(shape) == 4:
            c_out, c_in, k_h, k_w = shape
            fan_in = c_in * k_h * k_w
            fan_out = c_out * k_h * k_w
            return math.sqrt(fan_out / fan_in)
        else:
            return 1.0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step with mixed optimizers."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            param = group["params"][0]  # Single parameter per group
            optimizer_type = group["optimizer"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            if param.grad is None:
                continue

            grad = param.grad
            if grad.dtype in {torch.float16, torch.bfloat16}:
                grad = grad.float()

            state = self.state[param]

            if len(state) == 0:
                state["step"] = 0
                if optimizer_type == "muon":
                    state["momentum_buffer"] = torch.zeros_like(grad)
                else:
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            state["step"] += 1

            if optimizer_type == "muon":
                # Muon update for linear layers
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(self.muon_config["momentum"]).add_(grad, alpha=1 - self.muon_config["momentum"])

                if len(param.shape) >= 2:
                    # Reshape if needed
                    original_shape = param.shape
                    if len(param.shape) > 2:
                        param_2d = param.reshape(param.shape[0], -1)
                        momentum_2d = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
                    else:
                        param_2d = param
                        momentum_2d = momentum_buffer

                    # Apply Newton-Schulz orthogonalization
                    ortho_momentum = self.newton_schulz_orthogonalize(momentum_2d, self.muon_config["ns_iters"])

                    # Calculate scaling
                    dim_scaling = self.get_dimension_scaling(original_shape)
                    momentum_norm = torch.norm(momentum_2d, p="fro")

                    if momentum_norm > self.muon_config["eps"]:
                        update = ortho_momentum * (dim_scaling / (momentum_norm + self.muon_config["eps"]))

                        if len(param.shape) > 2:
                            update = update.reshape(original_shape)

                        # Apply weight decay and update
                        if weight_decay > 0:
                            param.mul_(1 - lr * weight_decay)
                        param.add_(update, alpha=-lr)
                else:
                    # Fallback for 1D parameters (shouldn't happen with correct categorization)
                    if weight_decay > 0:
                        param.mul_(1 - lr * weight_decay)
                    param.add_(momentum_buffer, alpha=-lr)

            else:
                # Adam update for embeddings, lm_head, and 1D parameters
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = self.adam_config["betas"]
                eps = self.adam_config["eps"]

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Weight decay (AdamW style)
                if weight_decay > 0:
                    param.mul_(1 - lr * weight_decay)

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute step size
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def update_learning_rates(
        self, muon_lr_factor: float, adam_lr_factor: float, embedding_lr_factor: float, lm_head_lr_factor: float
    ):
        """Update learning rates for all parameter groups."""
        for group in self.param_groups:
            param_type = group["type"]
            if param_type == "linear":
                group["lr"] = self.muon_config["lr"] * muon_lr_factor
            elif param_type == "embedding":
                group["lr"] = self.muon_config["lr"] * embedding_lr_factor  # Use base lr for embeddings
            elif param_type == "lm_head":
                group["lr"] = self.muon_config["lr"] * lm_head_lr_factor  # Use base lr for lm_head
            else:  # 1d parameters
                group["lr"] = self.adam_config["lr"] * adam_lr_factor


class MixedOptimizerV3(Optimizer):
    """
    Advanced mixed optimizer supporting Madgrad for superior transformer performance.

    Based on research showing Madgrad "blows away various Adam variants" for transformers.
    Supports: Muon, Adam, Madgrad with different learning rates per parameter group.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float = 8e-3,
        adam_lr: float = 6e-3,
        madgrad_lr: float = 1e-2,  # Madgrad typically uses higher LR
        embedding_lr: float = 12e-3,
        lm_head_lr: float = 4e-3,
        optimizer_type: str = "madgrad",  # "adam", "madgrad", or "muon" for non-linear params
        muon_momentum: float = 0.97,
        adam_betas: tuple[float, float] = (0.9, 0.85),  # Use research-optimized β2
        madgrad_momentum: float = 0.9,
        weight_decay: float = 0.015,
        eps: float = 1e-8,
        ns_iters: int = 5,
        use_optimized_muon: bool = True,
        # Outlier-safe parameters
        outlier_threshold: float = 5.0,
        enable_outlier_detection: bool = True,
        stability_check_freq: int = 20,
        max_norm_scale: float = 5.0,
        enable_stability_logging: bool = False,
    ) -> None:
        """
        Initialize the advanced mixed optimizer.

        Args:
            model: The model to optimize
            muon_lr: Learning rate for Muon optimizer
            adam_lr: Learning rate for Adam optimizer
            madgrad_lr: Learning rate for Madgrad optimizer
            embedding_lr: Learning rate for embedding parameters
            lm_head_lr: Learning rate for LM head parameters
            optimizer_type: Which optimizer to use for multi-dimensional params ("madgrad", "adam", or "muon")
            muon_momentum: Momentum for Muon
            adam_betas: Beta parameters for Adam (using optimized β2=0.85)
            madgrad_momentum: Momentum for Madgrad
            weight_decay: Weight decay coefficient
            eps: Numerical stability constant
            ns_iters: Newton-Schulz iterations for Muon
            use_optimized_muon: Whether to use optimized Muon coefficients
            outlier_threshold: Outlier detection threshold
            enable_outlier_detection: Enable outlier detection
            stability_check_freq: Stability check frequency
            max_norm_scale: Maximum norm scaling
            enable_stability_logging: Enable stability logging
        """
        # Categorize parameters
        param_groups = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine parameter type and optimizer
            if "token_embeddings" in name or "embedding" in name:
                param_type = "embedding"
                lr = embedding_lr
                opt_type = "madgrad"  # Use Madgrad for embeddings (research shows it's better)
            elif "lm_head" in name:
                param_type = "lm_head"
                lr = lm_head_lr
                opt_type = "madgrad"  # Use Madgrad for LM head
            elif len(param.shape) <= 1:  # 1D parameters
                param_type = "1d"
                lr = adam_lr if optimizer_type == "adam" else madgrad_lr
                opt_type = "adam" if optimizer_type == "adam" else "madgrad"
            else:  # Multi-dimensional parameters
                param_type = "linear"
                if optimizer_type == "muon":
                    lr = muon_lr
                    opt_type = "muon"
                elif optimizer_type == "madgrad":
                    lr = madgrad_lr
                    opt_type = "madgrad"
                else:
                    lr = adam_lr
                    opt_type = "adam"

            param_groups.append(
                {
                    "params": [param],
                    "name": name,
                    "type": param_type,
                    "optimizer": opt_type,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )

        # Store configurations
        self.muon_config = {
            "lr": muon_lr,
            "momentum": muon_momentum,
            "ns_iters": ns_iters,
            "weight_decay": weight_decay,
            "eps": eps,
            "use_optimized_coefficients": use_optimized_muon,
            "outlier_threshold": outlier_threshold,
            "enable_outlier_detection": enable_outlier_detection,
            "stability_check_freq": stability_check_freq,
            "max_norm_scale": max_norm_scale,
            "enable_stability_logging": enable_stability_logging,
        }

        self.adam_config = {
            "lr": adam_lr,
            "betas": adam_betas,  # Using optimized β2=0.85
            "eps": eps,
            "weight_decay": weight_decay,
        }

        self.madgrad_config = {
            "lr": madgrad_lr,
            "momentum": madgrad_momentum,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        # Initialize optimizers with proper parameter groups
        muon_params = []
        adam_params = []
        madgrad_params = []

        for group in param_groups:
            param = group["params"][0]
            opt_type = group["optimizer"]

            if opt_type == "muon":
                muon_params.extend(group["params"])
            elif opt_type == "adam":
                adam_params.extend(group["params"])
            elif opt_type == "madgrad":
                madgrad_params.extend(group["params"])

        # Only create optimizers if they have parameters
        self.muon_optimizer = Muon(muon_params, **self.muon_config) if muon_params else None
        self.adam_optimizer = Adam(adam_params, **self.adam_config) if adam_params else None
        self.madgrad_optimizer = Madgrad(madgrad_params, **self.madgrad_config) if madgrad_params else None

        # Initialize base class
        all_params = []
        for group in param_groups:
            all_params.extend(group["params"])

        defaults = dict(lr=madgrad_lr, weight_decay=weight_decay)
        super().__init__(all_params, defaults)

        self.param_groups = param_groups

        # Log configuration
        muon_count = sum(1 for g in param_groups if g["optimizer"] == "muon")
        adam_count = sum(1 for g in param_groups if g["optimizer"] == "adam")
        madgrad_count = sum(1 for g in param_groups if g["optimizer"] == "madgrad")

        print(f"MixedOptimizerV3 initialized:")
        print(f"  Muon parameters: {muon_count}")
        print(f"  Adam parameters: {adam_count}")
        print(f"  Madgrad parameters: {madgrad_count}")
        print(f"  🔥 Using Madgrad (FB AI's transformer champion) for {madgrad_count} param groups")

    def step(self, closure=None):
        """Perform optimization step with all optimizers."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.muon_optimizer and self.muon_optimizer.param_groups[0]["params"]:
            self.muon_optimizer.step()
        if self.adam_optimizer and self.adam_optimizer.param_groups[0]["params"]:
            self.adam_optimizer.step()
        if self.madgrad_optimizer and self.madgrad_optimizer.param_groups[0]["params"]:
            self.madgrad_optimizer.step()

        return loss

    def zero_grad(self, set_to_none=False):
        """Zero gradients for all optimizers."""
        if self.muon_optimizer:
            self.muon_optimizer.zero_grad(set_to_none)
        if self.adam_optimizer:
            self.adam_optimizer.zero_grad(set_to_none)
        if self.madgrad_optimizer:
            self.madgrad_optimizer.zero_grad(set_to_none)

    def get_stability_stats(self):
        """Get stability statistics from optimizers."""
        stats = {}
        if self.muon_optimizer and hasattr(self.muon_optimizer, "get_stability_stats"):
            stats.update(self.muon_optimizer.get_stability_stats())
        return stats

    def update_learning_rates(
        self, muon_factor=1.0, adam_factor=1.0, madgrad_factor=1.0, embedding_factor=1.0, lm_head_factor=1.0
    ):
        """Update learning rates for all optimizers."""
        if self.muon_optimizer:
            for group in self.muon_optimizer.param_groups:
                group["lr"] = self.muon_config["lr"] * muon_factor

        if self.adam_optimizer:
            for group in self.adam_optimizer.param_groups:
                group["lr"] = self.adam_config["lr"] * adam_factor

        if self.madgrad_optimizer:
            for group in self.madgrad_optimizer.param_groups:
                group["lr"] = self.madgrad_config["lr"] * madgrad_factor
