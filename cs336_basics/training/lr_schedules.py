"""
Advanced learning rate scheduling utilities for optimal training.
"""

from __future__ import annotations

import math


def cosine_learning_rate_schedule(
    iteration: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    """
    Cosine annealing learning rate schedule with linear warmup.

    This implements the cosine learning rate schedule used in modern language models.
    The schedule has three phases:
    1. Linear warmup: linearly increase from 0 to max_learning_rate
    2. Cosine annealing: cosine decay from max_learning_rate to min_learning_rate
    3. Post-annealing: constant at min_learning_rate

    Args:
        iteration: current iteration number
        max_learning_rate: maximum learning rate (at end of warmup)
        min_learning_rate: minimum learning rate (at end of cosine annealing)
        warmup_iters: number of warmup iterations
        cosine_cycle_iters: total number of iterations for cosine annealing

    Returns:
        learning rate for the current iteration
    """
    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters
    elif iteration <= cosine_cycle_iters:
        progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + (max_learning_rate - min_learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        return min_learning_rate


def aggressive_cosine_schedule(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    total_iters: int,
    peak_ratio: float = 0.15,
    fast_decay_ratio: float = 0.4,
) -> float:
    """
    Aggressive cosine learning rate schedule optimized for fast convergence within time constraints.

    This schedule is designed to:
    1. Warm up very quickly to high learning rates
    2. Maintain high learning rates for rapid initial progress
    3. Use fast decay to quickly reach stable convergence
    4. Fine-tune with lower learning rates for final performance

    Args:
        iteration: current iteration number
        max_learning_rate: maximum learning rate
        min_learning_rate: minimum learning rate
        warmup_iters: number of warmup iterations (should be small)
        total_iters: total number of training iterations
        peak_ratio: ratio of training spent at near-peak learning rates
        fast_decay_ratio: ratio of training spent in fast decay phase

    Returns:
        learning rate for the current iteration
    """
    if iteration < warmup_iters:
        if iteration == 0:
            return max_learning_rate * 0.01

        warmup_progress = iteration / warmup_iters
        warmup_factor = 1 - math.exp(-4 * warmup_progress)

        baseline_lr = max_learning_rate * 0.005
        warmup_lr = max_learning_rate * warmup_factor

        return max(baseline_lr, warmup_lr)

    remaining_iters = total_iters - warmup_iters
    post_warmup_iter = iteration - warmup_iters

    peak_iters = int(remaining_iters * peak_ratio)
    if post_warmup_iter < peak_iters:
        modulation = 1.0 - 0.05 * (1 - math.cos(2 * math.pi * post_warmup_iter / peak_iters))
        return max_learning_rate * modulation

    fast_decay_iters = int(remaining_iters * fast_decay_ratio)
    if post_warmup_iter < peak_iters + fast_decay_iters:
        decay_progress = (post_warmup_iter - peak_iters) / fast_decay_iters
        decay_factor = 0.5 * (1 + math.cos(math.pi * decay_progress))
        target_lr = min_learning_rate + 0.3 * (max_learning_rate - min_learning_rate)
        return target_lr + (max_learning_rate - target_lr) * decay_factor

    fine_tune_start = peak_iters + fast_decay_iters
    fine_tune_progress = (post_warmup_iter - fine_tune_start) / (remaining_iters - fine_tune_start)
    fine_tune_progress = min(fine_tune_progress, 1.0)

    start_lr = min_learning_rate + 0.3 * (max_learning_rate - min_learning_rate)
    final_decay_factor = 0.5 * (1 + math.cos(math.pi * fine_tune_progress))
    return min_learning_rate + (start_lr - min_learning_rate) * final_decay_factor


def improved_cosine_schedule(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    total_iters: int,
    restart_factor: float = 0.0,
) -> float:
    """
    Improved cosine learning rate schedule with better warmup and optional restarts.

    Features:
    - Smoother warmup transition
    - More stable convergence
    - Optional cosine restarts

    Args:
        iteration: current iteration number
        max_learning_rate: maximum learning rate
        min_learning_rate: minimum learning rate
        warmup_iters: number of warmup iterations
        total_iters: total number of training iterations
        restart_factor: factor for cosine restarts (0.0 = no restarts)

    Returns:
        learning rate for the current iteration
    """
    if iteration < warmup_iters:
        warmup_factor = 0.5 * (1 + math.cos(math.pi * (1 - iteration / warmup_iters)))
        return max_learning_rate * (1 - warmup_factor)
    else:
        progress = (iteration - warmup_iters) / (total_iters - warmup_iters)

        if restart_factor > 0.0:
            restart_period = int((total_iters - warmup_iters) * restart_factor)
            if restart_period > 0:
                progress = progress % (restart_period / (total_iters - warmup_iters))
                progress = progress * (total_iters - warmup_iters) / restart_period

        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_factor


def exponential_decay_schedule(
    iteration: int,
    initial_lr: float,
    decay_rate: float,
    decay_steps: int,
    staircase: bool = False,
) -> float:
    """
    Exponential decay learning rate schedule.

    Args:
        iteration: current iteration number
        initial_lr: initial learning rate
        decay_rate: decay rate (e.g., 0.96)
        decay_steps: steps between decay
        staircase: whether to use staircase decay

    Returns:
        decayed learning rate
    """
    if staircase:
        decay_factor = decay_rate ** (iteration // decay_steps)
    else:
        decay_factor = decay_rate ** (iteration / decay_steps)

    return initial_lr * decay_factor


def polynomial_decay_schedule(
    iteration: int,
    initial_lr: float,
    min_lr: float,
    total_steps: int,
    power: float = 1.0,
) -> float:
    """
    Polynomial decay learning rate schedule.

    Args:
        iteration: current iteration number
        initial_lr: initial learning rate
        min_lr: minimum learning rate
        total_steps: total training steps
        power: polynomial power (1.0 = linear decay)

    Returns:
        decayed learning rate
    """
    if iteration >= total_steps:
        return min_lr

    decay_factor = (1 - iteration / total_steps) ** power
    return min_lr + (initial_lr - min_lr) * decay_factor


def one_cycle_schedule(
    iteration: int,
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.3,
    anneal_strategy: str = "cos",
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> float:
    """
    One-cycle learning rate schedule popular for fast convergence.

    Args:
        iteration: current iteration number
        max_lr: maximum learning rate
        total_steps: total training steps
        pct_start: percentage of cycle spent increasing learning rate
        anneal_strategy: annealing strategy ('cos' or 'linear')
        div_factor: factor for initial learning rate
        final_div_factor: factor for final learning rate

    Returns:
        learning rate for current iteration
    """
    initial_lr = max_lr / div_factor
    final_lr = initial_lr / final_div_factor

    step_up = int(total_steps * pct_start)
    step_down = total_steps - step_up

    if iteration <= step_up:
        return initial_lr + (max_lr - initial_lr) * iteration / step_up
    else:
        progress = (iteration - step_up) / step_down

        if anneal_strategy == "cos":
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        else:
            factor = 1 - progress

        return final_lr + (max_lr - final_lr) * factor


def warmup_then_constant_schedule(
    iteration: int,
    target_lr: float,
    warmup_iters: int,
    warmup_type: str = "linear",
) -> float:
    """
    Warmup to target learning rate then keep constant.

    Args:
        iteration: current iteration number
        target_lr: target learning rate after warmup
        warmup_iters: number of warmup iterations
        warmup_type: type of warmup ('linear' or 'cosine')

    Returns:
        learning rate for current iteration
    """
    if iteration < warmup_iters:
        if warmup_type == "linear":
            return target_lr * iteration / warmup_iters
        elif warmup_type == "cosine":
            factor = 0.5 * (1 - math.cos(math.pi * iteration / warmup_iters))
            return target_lr * factor
        else:
            raise ValueError(f"Unknown warmup_type: {warmup_type}")
    else:
        return target_lr


def get_scheduler(
    scheduler_type: str, max_lr: float, min_lr: float = None, warmup_steps: int = 0, total_steps: int = None, **kwargs
) -> callable:
    """
    Factory function to get learning rate scheduler.

    Args:
        scheduler_type: type of scheduler
        max_lr: maximum learning rate
        min_lr: minimum learning rate
        warmup_steps: warmup steps
        total_steps: total training steps
        **kwargs: additional scheduler-specific arguments

    Returns:
        scheduler function that takes iteration and returns learning rate
    """
    if min_lr is None:
        min_lr = max_lr * 0.1

    if scheduler_type == "cosine":

        def scheduler(iteration):
            return cosine_learning_rate_schedule(
                iteration, max_lr, min_lr, warmup_steps, total_steps or warmup_steps * 10
            )
    elif scheduler_type == "aggressive":

        def scheduler(iteration):
            return aggressive_cosine_schedule(
                iteration, max_lr, min_lr, warmup_steps, total_steps or warmup_steps * 10, **kwargs
            )
    elif scheduler_type == "improved_cosine":

        def scheduler(iteration):
            return improved_cosine_schedule(
                iteration, max_lr, min_lr, warmup_steps, total_steps or warmup_steps * 10, **kwargs
            )
    elif scheduler_type == "one_cycle":

        def scheduler(iteration):
            return one_cycle_schedule(iteration, max_lr, total_steps or warmup_steps * 10, **kwargs)
    elif scheduler_type == "constant":

        def scheduler(iteration):
            return warmup_then_constant_schedule(iteration, max_lr, warmup_steps, **kwargs)
    elif scheduler_type == "polynomial":

        def scheduler(iteration):
            return polynomial_decay_schedule(iteration, max_lr, min_lr, total_steps or warmup_steps * 10, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    return scheduler
