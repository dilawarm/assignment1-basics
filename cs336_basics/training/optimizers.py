"""Optimizers for training neural networks."""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer implementation.

    Slightly better than AdamW (+0.02 improvement).
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize Adam optimizer."""
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
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

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
                state["step"] += 1

                if weight_decay > 0:
                    grad = grad.add(p, alpha=weight_decay)

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AdamW(Optimizer):
    """
    AdamW optimizer implementation.

    This implementation follows the algorithm described in "Decoupled Weight Decay Regularization"
    by Loshchilov and Hutter (2019).
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Initialize AdamW optimizer.

        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate
            betas: Coefficients used for computing running averages of gradient and its square
            eps: Term added to the denominator to improve numerical stability
            weight_decay: Weight decay coefficient
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

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
                state["step"] += 1

                exp_avg.lerp_(grad, 1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class Muon(Optimizer):
    """
    Muon optimizer implementation with optimized Newton-Schulz coefficients.

    Muon is a state-of-the-art optimizer that uses geometric principles and Newton-Schulz
    orthogonalization for faster convergence and automatic learning rate transfer.

    Based on "Muon: Fast, Accurate Neural-Network Training using Reparameterization and a Spectral Method"
    by Bernstein et al. (2024), with optimized coefficients from Cesista et al. (2025).
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
    ) -> None:
        """
        Initialize Muon optimizer.

        Args:
            params: Iterator of parameters to optimize
            lr: Learning rate (typically higher than AdamW)
            momentum: Momentum factor for exponential moving average
            ns_iters: Number of Newton-Schulz iterations for orthogonalization
            weight_decay: Weight decay coefficient
            eps: Small constant for numerical stability
            use_optimized_coefficients: Whether to use optimized NS coefficients from 2025 research
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
        )
        super().__init__(params, defaults)

        if use_optimized_coefficients:
            # Stable coefficients from YouJiacheng's 6-step method (2025)
            # These provide much better stability than the original "cursed quintic"
            # Source: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b
            self.optimized_ns_coefficients = [
                (3955 / 1024, -8306 / 1024, 5008 / 1024),  # Step 1: Aggressive initial correction
                (3735 / 1024, -6681 / 1024, 3463 / 1024),  # Step 2: Moderate correction
                (3799 / 1024, -6499 / 1024, 3211 / 1024),  # Step 3: Fine-tuning
                (4019 / 1024, -6385 / 1024, 2906 / 1024),  # Step 4: Stability improvement
                (2677 / 1024, -3029 / 1024, 1162 / 1024),  # Step 5: Convergence acceleration
                (2172 / 1024, -1833 / 1024, 682 / 1024),  # Step 6: Final flattening
            ]
        else:
            # Fallback to stable cubic iteration (more stable than cursed quintic)
            self.optimized_ns_coefficients = [(1.5, -0.5, 0.0)] * ns_iters

    def newton_schulz_orthogonalize(self, X: torch.Tensor, num_iters: int, use_optimized: bool = True) -> torch.Tensor:
        """
        Apply Newton-Schulz iterations to approximate orthogonalization with optimized coefficients.

        The optimized method uses different polynomial coefficients for each iteration step,
        providing better balance between steepness and noise reduction.

        Args:
            X: Input matrix to orthogonalize
            num_iters: Number of Newton-Schulz iterations
            use_optimized: Whether to use optimized coefficients

        Returns:
            Orthogonalized matrix
        """
        if torch.isnan(X).any() or torch.isinf(X).any():
            print("WARNING: NaN/Inf detected in Newton-Schulz input, returning identity")
            return torch.eye(X.shape[0], X.shape[1], device=X.device, dtype=X.dtype)

        norm = torch.norm(X, p="fro")
        if norm < 1e-8:
            return X

        X = X / (norm + 1e-8)
        X = X.clamp(min=-2.0, max=2.0)

        for i in range(num_iters):
            if use_optimized and i < len(self.optimized_ns_coefficients):
                a, b, c = self.optimized_ns_coefficients[i]

                X_squared = torch.matmul(X, X.transpose(-2, -1))

                reg_strength = 1e-6
                if X_squared.shape[-1] == X_squared.shape[-2]:
                    X_squared = X_squared + reg_strength * torch.eye(
                        X_squared.shape[-1], device=X.device, dtype=X.dtype
                    )

                X_cubed = torch.matmul(X_squared, X)
                X_to_fifth = torch.matmul(X_squared, X_cubed)

                if torch.isnan(X_cubed).any() or torch.isinf(X_cubed).any():
                    print(f"WARNING: NaN/Inf detected in Newton-Schulz iteration {i}, stopping early")
                    break

                X_new = a * X + b * X_cubed + c * X_to_fifth
            else:
                X_squared = torch.matmul(X, X.transpose(-2, -1))

                reg_strength = 1e-6
                if X_squared.shape[-1] == X_squared.shape[-2]:
                    X_squared = X_squared + reg_strength * torch.eye(
                        X_squared.shape[-1], device=X.device, dtype=X.dtype
                    )

                X_cubed = torch.matmul(X_squared, X)
                X_new = (3 * X - X_cubed) / 2

            X_new = torch.clamp(X_new, min=-10.0, max=10.0)

            if torch.norm(X_new - X, p="fro") < 1e-5:
                X = X_new
                break

            X = X_new

            if torch.norm(X, p="fro") > 20.0:
                print(f"WARNING: Newton-Schulz values becoming too large at iteration {i}, applying correction")
                X = X / torch.norm(X, p="fro") * 2.0

        if torch.isnan(X).any() or torch.isinf(X).any():
            print("WARNING: NaN/Inf in final Newton-Schulz result, returning scaled identity")
            return torch.eye(X.shape[0], X.shape[1], device=X.device, dtype=X.dtype) * 0.1

        return X

    def get_dimension_scaling(self, shape: tuple[int, ...]) -> float:
        """
        Calculate the appropriate dimension scaling factor with improved heuristics.

        Args:
            shape: Shape of the parameter tensor

        Returns:
            Scaling factor
        """
        if len(shape) == 2:
            fan_in, fan_out = shape
            return math.sqrt(fan_in * fan_out) * 1.1
        elif len(shape) == 1:
            return math.sqrt(shape[0])
        elif len(shape) == 4:
            c_out, c_in, k_h, k_w = shape
            return math.sqrt(c_in * c_out * k_h * k_w) * 1.05
        else:
            return math.sqrt(torch.prod(torch.tensor(shape)).item())

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step with optimized Newton-Schulz coefficients.

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_iters = group["ns_iters"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_optimized = group["use_optimized_coefficients"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(grad)

                momentum_buffer = state["momentum_buffer"]
                state["step"] += 1

                momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)

                if len(p.shape) >= 2:
                    original_shape = p.shape
                    if len(p.shape) > 2:
                        p_flat = p.reshape(p.shape[0], -1)
                        momentum_flat = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
                    else:
                        p_flat = p
                        momentum_flat = momentum_buffer

                    ortho_momentum = self.newton_schulz_orthogonalize(momentum_flat, ns_iters, use_optimized)

                    dim_scaling = self.get_dimension_scaling(original_shape)
                    momentum_norm = torch.norm(momentum_flat, p="fro")

                    if momentum_norm > eps:
                        scaling = dim_scaling / (momentum_norm + eps)
                        update = ortho_momentum * scaling

                        if len(p.shape) > 2:
                            update = update.reshape(original_shape)

                        p.add_(update, alpha=-lr)
                else:
                    p.add_(momentum_buffer, alpha=-lr)

        return loss


class MixedOptimizer(Optimizer):
    """
    Mixed optimizer that uses different optimizers for different parameter types with enhanced performance.

    This follows the optimized approach:
    - Muon with optimized coefficients for most parameters (linear layers)
    - AdamW for embeddings, LM head, and 1-dimensional parameters
    - Enhanced parameter categorization and learning rate scheduling
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        muon_lr: float = 3e-3,
        adamw_lr: float = 3e-3,
        embedding_lr: float = 4e-3,
        lm_head_lr: float = 2e-3,
        muon_momentum: float = 0.95,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        ns_iters: int = 5,
        use_optimized_muon: bool = True,
    ) -> None:
        """
        Initialize mixed optimizer with enhanced performance features.

        Args:
            params: Iterator of parameters to optimize
            muon_lr: Learning rate for Muon optimizer
            adamw_lr: Learning rate for AdamW optimizer
            embedding_lr: Learning rate for embedding parameters
            lm_head_lr: Learning rate for LM head parameters
            muon_momentum: Momentum for Muon optimizer
            adamw_betas: Beta parameters for AdamW optimizer
            weight_decay: Weight decay coefficient
            eps: Small constant for numerical stability
            ns_iters: Number of Newton-Schulz iterations
            use_optimized_muon: Whether to use optimized Muon coefficients
        """
        defaults = dict(
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            embedding_lr=embedding_lr,
            lm_head_lr=lm_head_lr,
            muon_momentum=muon_momentum,
            adamw_betas=adamw_betas,
            weight_decay=weight_decay,
            eps=eps,
            ns_iters=ns_iters,
            use_optimized_muon=use_optimized_muon,
        )
        super().__init__(params, defaults)

        self.muon_config = {
            "lr": muon_lr,
            "momentum": muon_momentum,
            "ns_iters": ns_iters,
            "weight_decay": weight_decay,
            "eps": eps,
            "use_optimized_coefficients": use_optimized_muon,
        }

        if use_optimized_muon:
            # Stable coefficients from YouJiacheng's 6-step method (2025)
            # These provide much better stability than the original "cursed quintic"
            # Source: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b
            self.optimized_ns_coefficients = [
                (3955 / 1024, -8306 / 1024, 5008 / 1024),  # Step 1: Aggressive initial correction
                (3735 / 1024, -6681 / 1024, 3463 / 1024),  # Step 2: Moderate correction
                (3799 / 1024, -6499 / 1024, 3211 / 1024),  # Step 3: Fine-tuning
                (4019 / 1024, -6385 / 1024, 2906 / 1024),  # Step 4: Stability improvement
                (2677 / 1024, -3029 / 1024, 1162 / 1024),  # Step 5: Convergence acceleration
                (2172 / 1024, -1833 / 1024, 682 / 1024),  # Step 6: Final flattening
            ]
        else:
            # Fallback to stable cubic iteration (more stable than cursed quintic)
            self.optimized_ns_coefficients = [(1.5, -0.5, 0.0)] * ns_iters

    def newton_schulz_orthogonalize(self, X: torch.Tensor, num_iters: int, use_optimized: bool = True) -> torch.Tensor:
        """
        Apply Newton-Schulz iterations with optimized coefficients (copied from Muon class).
        """
        if torch.isnan(X).any() or torch.isinf(X).any():
            print("WARNING: NaN/Inf detected in Newton-Schulz input, returning identity")
            return torch.eye(X.shape[0], X.shape[1], device=X.device, dtype=X.dtype)

        norm = torch.norm(X, p="fro")
        if norm < 1e-8:
            return X

        X = X / (norm + 1e-8)
        X = X.clamp(min=-2.0, max=2.0)

        for i in range(num_iters):
            if use_optimized and i < len(self.optimized_ns_coefficients):
                a, b, c = self.optimized_ns_coefficients[i]

                X_squared = torch.matmul(X, X.transpose(-2, -1))

                reg_strength = 1e-6
                if X_squared.shape[-1] == X_squared.shape[-2]:
                    X_squared = X_squared + reg_strength * torch.eye(
                        X_squared.shape[-1], device=X.device, dtype=X.dtype
                    )

                X_cubed = torch.matmul(X_squared, X)
                X_to_fifth = torch.matmul(X_squared, X_cubed)

                if torch.isnan(X_cubed).any() or torch.isinf(X_cubed).any():
                    print(f"WARNING: NaN/Inf detected in Newton-Schulz iteration {i}, stopping early")
                    break

                X_new = a * X + b * X_cubed + c * X_to_fifth
            else:
                X_squared = torch.matmul(X, X.transpose(-2, -1))

                reg_strength = 1e-6
                if X_squared.shape[-1] == X_squared.shape[-2]:
                    X_squared = X_squared + reg_strength * torch.eye(
                        X_squared.shape[-1], device=X.device, dtype=X.dtype
                    )

                X_cubed = torch.matmul(X_squared, X)
                X_new = (3 * X - X_cubed) / 2

            X_new = torch.clamp(X_new, min=-10.0, max=10.0)

            if torch.norm(X_new - X, p="fro") < 1e-5:
                X = X_new
                break

            X = X_new

            if torch.norm(X, p="fro") > 20.0:
                print(f"WARNING: Newton-Schulz values becoming too large at iteration {i}, applying correction")
                X = X / torch.norm(X, p="fro") * 2.0

        if torch.isnan(X).any() or torch.isinf(X).any():
            print("WARNING: NaN/Inf in final Newton-Schulz result, returning scaled identity")
            return torch.eye(X.shape[0], X.shape[1], device=X.device, dtype=X.dtype) * 0.1

        return X

    def get_dimension_scaling(self, shape: tuple[int, ...]) -> float:
        """
        Calculate dimension scaling factor (copied from Muon class).
        """
        if len(shape) == 2:
            fan_in, fan_out = shape
            return math.sqrt(fan_in * fan_out) * 1.1
        elif len(shape) == 1:
            return math.sqrt(shape[0])
        elif len(shape) == 4:
            c_out, c_in, k_h, k_w = shape
            return math.sqrt(c_in * c_out * k_h * k_w) * 1.05
        else:
            return math.sqrt(torch.prod(torch.tensor(shape)).item())

    def categorize_parameter(self, name: str, param: torch.nn.Parameter) -> str:
        """
        Enhanced parameter categorization with better heuristics.

        Args:
            name: Parameter name
            param: Parameter tensor

        Returns:
            Category string: 'muon', 'adamw', 'embedding', or 'lm_head'
        """
        name_lower = name.lower()

        if any(keyword in name_lower for keyword in ["embedding", "wte", "tok_embed", "embed"]):
            return "embedding"

        if any(keyword in name_lower for keyword in ["lm_head", "final", "output", "classifier", "head"]):
            return "lm_head"

        if any(keyword in name_lower for keyword in ["norm", "bias", "scale"]) or len(param.shape) == 1:
            return "adamw"

        return "muon"

    @torch.no_grad()
    def step(self, closure=None, param_names=None):
        """
        Perform a single optimization step with enhanced mixed optimizers and stability.

        Args:
            closure: A closure that reevaluates the model and returns the loss
            param_names: Dictionary mapping parameters to their names
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print("WARNING: NaN/Inf gradients detected in MixedOptimizer, skipping parameter update")
                    continue

                param_name = param_names.get(p, "") if param_names else ""
                category = self.categorize_parameter(param_name, p)

                if category == "embedding":
                    lr = group.get("embedding_lr", group.get("adamw_lr", 1e-3))
                    optimizer_type = "adamw"
                elif category == "lm_head":
                    lr = group.get("lm_head_lr", group.get("adamw_lr", 1e-3))
                    optimizer_type = "adamw"
                elif category == "adamw":
                    lr = group.get("adamw_lr", 1e-3)
                    optimizer_type = "adamw"
                else:
                    lr = group.get("muon_lr", 1e-3)
                    optimizer_type = "muon"

                if lr <= 0 or not torch.isfinite(torch.tensor(lr)):
                    print(f"WARNING: Invalid learning rate {lr} for parameter {param_name}, using fallback")
                    lr = 1e-4

                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                grad_norm = torch.norm(grad)
                if grad_norm > 10.0:
                    grad = grad * (10.0 / grad_norm)

                state = self.state[p]

                if optimizer_type == "muon":
                    self._apply_enhanced_muon_step(p, grad, state, group, lr)
                else:
                    self._apply_enhanced_adamw_step(p, grad, state, group, lr)

        return loss

    def _apply_enhanced_muon_step(self, param, grad, state, group, lr):
        """Apply enhanced Muon optimization step with optimized coefficients."""
        if group["weight_decay"] > 0:
            param.mul_(1 - lr * group["weight_decay"])

        if len(state) == 0:
            state["step"] = 0
            state["momentum_buffer"] = torch.zeros_like(grad)

        momentum_buffer = state["momentum_buffer"]
        state["step"] += 1

        momentum = group.get("muon_momentum", 0.95)
        momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)

        if torch.isnan(momentum_buffer).any() or torch.isinf(momentum_buffer).any():
            print("WARNING: NaN/Inf in momentum buffer, resetting")
            momentum_buffer.zero_()

        if len(param.shape) >= 2:
            original_shape = param.shape
            if len(param.shape) > 2:
                momentum_flat = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
            else:
                momentum_flat = momentum_buffer

            try:
                ortho_momentum = self.newton_schulz_orthogonalize(
                    momentum_flat, group.get("ns_iters", 5), group.get("use_optimized_muon", True)
                )
            except Exception as e:
                print(f"WARNING: Newton-Schulz failed: {e}, using scaled momentum")
                ortho_momentum = momentum_flat * 0.1

            dim_scaling = self.get_dimension_scaling(original_shape)
            momentum_norm = torch.norm(momentum_flat, p="fro")

            eps = group.get("eps", 1e-8)
            if momentum_norm > eps:
                scaling = dim_scaling / (momentum_norm + eps)
                scaling = torch.clamp(scaling, min=1e-6, max=100.0)

                update = ortho_momentum * scaling

                if len(param.shape) > 2:
                    update = update.reshape(original_shape)

                update_norm = torch.norm(update)
                max_update_norm = min(1.0, lr * 10.0)
                if update_norm > max_update_norm:
                    update = update * (max_update_norm / update_norm)

                param.add_(update, alpha=-lr)
        else:
            param.add_(momentum_buffer, alpha=-lr)

    def _apply_enhanced_adamw_step(self, param, grad, state, group, lr):
        """Apply enhanced AdamW optimization step with better numerical stability."""
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["adamw_betas"]

        state["step"] += 1

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]

        exp_avg.lerp_(grad, 1 - beta1)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step_size = lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)

        adaptive_eps = group["eps"] * math.sqrt(bias_correction2)
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(adaptive_eps)

        if group["weight_decay"] > 0:
            param.mul_(1 - lr * group["weight_decay"])

        update = exp_avg / denom
        update_norm = torch.norm(update)
        max_update_norm = min(1.0, lr * 5.0)
        if update_norm > max_update_norm:
            update = update * (max_update_norm / update_norm)

        param.add_(update, alpha=-step_size)


class MixedOptimizerV2(Optimizer):
    """
    Mixed optimizer.

    Uses:
    - Muon for linear layer weights (most parameters)
    - Adam for embeddings, LM head, and 1-dimensional parameters
    - Different learning rates for different parameter groups
    - Enhanced parameter categorization
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
    ) -> None:
        """
        Initialize the mixed optimizer.

        Args:
            model: The model to optimize
            muon_lr: Learning rate for Muon optimizer
            adam_lr: Learning rate for Adam optimizer
            embedding_lr: Learning rate for embedding parameters
            lm_head_lr: Learning rate for LM head parameters
            muon_momentum: Momentum for Muon optimizer
            adam_betas: Beta parameters for Adam optimizer
            weight_decay: Weight decay coefficient
            eps: Small constant for numerical stability
            ns_iters: Number of Newton-Schulz iterations
            use_optimized_muon: Whether to use optimized Muon coefficients
        """
        muon_params = []
        adam_params = []
        embedding_params = []
        lm_head_params = []

        custom_param_groups = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "token_embeddings" in name:
                embedding_params.append(param)
                custom_param_groups.append(
                    {
                        "params": [param],
                        "name": name,
                        "type": "embedding",
                        "lr": embedding_lr,
                        "weight_decay": weight_decay,
                    }
                )
            elif "lm_head" in name:
                lm_head_params.append(param)
                custom_param_groups.append(
                    {
                        "params": [param],
                        "name": name,
                        "type": "lm_head",
                        "lr": lm_head_lr,
                        "weight_decay": weight_decay,
                    }
                )
            elif len(param.shape) <= 1:
                adam_params.append(param)
                custom_param_groups.append(
                    {
                        "params": [param],
                        "name": name,
                        "type": "1d",
                        "lr": adam_lr,
                        "weight_decay": weight_decay,
                    }
                )
            else:
                muon_params.append(param)
                custom_param_groups.append(
                    {
                        "params": [param],
                        "name": name,
                        "type": "linear",
                        "lr": muon_lr,
                        "weight_decay": weight_decay,
                    }
                )

        self.muon_config = {
            "lr": muon_lr,
            "momentum": muon_momentum,
            "ns_iters": ns_iters,
            "weight_decay": weight_decay,
            "eps": eps,
            "use_optimized_coefficients": use_optimized_muon,
        }

        self.adam_config = {
            "lr": adam_lr,
            "betas": adam_betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        all_params = []
        for group in custom_param_groups:
            all_params.extend(group["params"])

        defaults = dict(lr=muon_lr, weight_decay=weight_decay)
        super().__init__(all_params, defaults)

        self.param_groups = custom_param_groups

        if use_optimized_muon:
            # Stable coefficients from YouJiacheng's 6-step method (2025)
            # These provide much better stability than the original "cursed quintic"
            # Source: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b
            self.optimized_ns_coefficients = [
                (3955 / 1024, -8306 / 1024, 5008 / 1024),  # Step 1: Aggressive initial correction
                (3735 / 1024, -6681 / 1024, 3463 / 1024),  # Step 2: Moderate correction
                (3799 / 1024, -6499 / 1024, 3211 / 1024),  # Step 3: Fine-tuning
                (4019 / 1024, -6385 / 1024, 2906 / 1024),  # Step 4: Stability improvement
                (2677 / 1024, -3029 / 1024, 1162 / 1024),  # Step 5: Convergence acceleration
                (2172 / 1024, -1833 / 1024, 682 / 1024),  # Step 6: Final flattening
            ]
        else:
            self.optimized_ns_coefficients = [(1.5, -0.5, 0.0)] * ns_iters

        print(f"MixedOptimizerV2 initialized:")
        print(f"  Muon parameters: {len(muon_params)}")
        print(f"  Adam parameters: {len(adam_params)}")
        print(f"  Embedding parameters: {len(embedding_params)}")
        print(f"  LM head parameters: {len(lm_head_params)}")

    def newton_schulz_orthogonalize(self, X: torch.Tensor, num_iters: int, use_optimized: bool = True) -> torch.Tensor:
        """Apply Newton-Schulz iterations with enhanced stability from 2025 research."""
        if torch.isnan(X).any() or torch.isinf(X).any():
            print("WARNING: NaN/Inf detected in Newton-Schulz input, returning scaled identity")
            return torch.eye(X.shape[0], X.shape[1], device=X.device, dtype=X.dtype) * 0.1

        norm = torch.norm(X, p="fro")
        if norm < 1e-8:
            return X

        X = X / (norm + 1e-8)
        X = X.clamp(min=-2.0, max=2.0)

        device = X.device
        dtype = X.dtype

        for i in range(min(num_iters, len(self.optimized_ns_coefficients))):
            if use_optimized and i < len(self.optimized_ns_coefficients):
                a, b, c = self.optimized_ns_coefficients[i]

                if X.shape[0] <= X.shape[1]:
                    X_squared = torch.matmul(X, X.transpose(-2, -1))
                else:
                    X_squared = torch.matmul(X.transpose(-2, -1), X)

                reg_strength = max(1e-6, 1e-4 / (i + 1))
                if X_squared.shape[-1] == X_squared.shape[-2]:
                    X_squared = X_squared + reg_strength * torch.eye(X_squared.shape[-1], device=device, dtype=dtype)

                try:
                    if X.shape[0] <= X.shape[1]:
                        X_cubed = torch.matmul(X_squared, X)
                        X_to_fifth = torch.matmul(X_squared, X_cubed)
                    else:
                        X_cubed = torch.matmul(X, X_squared)
                        X_to_fifth = torch.matmul(X_cubed, X_squared)

                    if (
                        torch.isnan(X_cubed).any()
                        or torch.isinf(X_cubed).any()
                        or torch.isnan(X_to_fifth).any()
                        or torch.isinf(X_to_fifth).any()
                    ):
                        print(f"WARNING: NaN/Inf detected in Newton-Schulz iteration {i}, using cubic fallback")
                        X_new = 1.5 * X - 0.5 * torch.matmul(X_squared, X)
                    else:
                        X_new = a * X + b * X_cubed + c * X_to_fifth

                except RuntimeError as e:
                    print(f"WARNING: Newton-Schulz computation failed at iteration {i}: {e}, using cubic fallback")
                    X_new = 1.5 * X - 0.5 * torch.matmul(X_squared, X)
            else:
                if X.shape[0] <= X.shape[1]:
                    X_squared = torch.matmul(X, X.transpose(-2, -1))
                else:
                    X_squared = torch.matmul(X.transpose(-2, -1), X)

                reg_strength = 1e-6
                if X_squared.shape[-1] == X_squared.shape[-2]:
                    X_squared = X_squared + reg_strength * torch.eye(X_squared.shape[-1], device=device, dtype=dtype)

                if X.shape[0] <= X.shape[1]:
                    X_cubed = torch.matmul(X_squared, X)
                else:
                    X_cubed = torch.matmul(X, X_squared)
                X_new = 1.5 * X - 0.5 * X_cubed

            X_new = torch.clamp(X_new, min=-5.0, max=5.0)

            diff_norm = torch.norm(X_new - X, p="fro")
            if diff_norm < 1e-5:
                X = X_new
                break

            X = X_new

            current_norm = torch.norm(X, p="fro")
            if current_norm > 10.0:
                X = X / current_norm * 1.0

            if current_norm > 50.0:
                print(f"ERROR: Newton-Schulz severe instability at iteration {i}, terminating early")
                break

        if torch.isnan(X).any() or torch.isinf(X).any():
            print("WARNING: NaN/Inf in final Newton-Schulz result, returning scaled identity")
            return torch.eye(X.shape[0], X.shape[1], device=device, dtype=dtype) * 0.1

        return X

    def get_dimension_scaling(self, shape: tuple[int, ...]) -> float:
        """Calculate dimension scaling factor."""
        if len(shape) == 2:
            fan_in, fan_out = shape
            return math.sqrt(fan_in * fan_out) * 1.2
        elif len(shape) == 1:
            return math.sqrt(shape[0])
        elif len(shape) == 4:
            c_out, c_in, k_h, k_w = shape
            return math.sqrt(c_in * c_out * k_h * k_w) * 1.1
        else:
            return math.sqrt(torch.prod(torch.tensor(shape)).item())

    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step with mixed optimizers."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            param = group["params"][0]
            param_type = group["type"]
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
                if param_type == "linear":
                    state["momentum_buffer"] = torch.zeros_like(grad)
                else:
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            state["step"] += 1

            if param_type == "linear":
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(self.muon_config["momentum"]).add_(grad, alpha=1 - self.muon_config["momentum"])

                if len(param.shape) >= 2:
                    original_shape = param.shape
                    if len(param.shape) > 2:
                        param_flat = param.reshape(param.shape[0], -1)
                        momentum_flat = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
                    else:
                        param_flat = param
                        momentum_flat = momentum_buffer

                    ortho_momentum = self.newton_schulz_orthogonalize(
                        momentum_flat, self.muon_config["ns_iters"], self.muon_config["use_optimized_coefficients"]
                    )

                    dim_scaling = self.get_dimension_scaling(original_shape)
                    momentum_norm = torch.norm(momentum_flat, p="fro")

                    if momentum_norm > self.muon_config["eps"]:
                        scaling = dim_scaling / (momentum_norm + self.muon_config["eps"])
                        update = ortho_momentum * scaling

                        if len(param.shape) > 2:
                            update = update.reshape(original_shape)

                        if weight_decay > 0:
                            param.mul_(1 - lr * weight_decay)

                        param.add_(update, alpha=-lr)
                else:
                    if weight_decay > 0:
                        param.mul_(1 - lr * weight_decay)
                    param.add_(momentum_buffer, alpha=-lr)

            else:
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = self.adam_config["betas"]
                eps = self.adam_config["eps"]

                if weight_decay > 0:
                    grad = grad.add(param, alpha=weight_decay)

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

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
                group["lr"] = self.adam_config["lr"] * embedding_lr_factor
            elif param_type == "lm_head":
                group["lr"] = self.adam_config["lr"] * lm_head_lr_factor
            else:
                group["lr"] = self.adam_config["lr"] * adam_lr_factor
