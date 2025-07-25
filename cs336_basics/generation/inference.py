"""
Inference Script for Trained TinyStories Transformer

Usage Examples:
    # Single completion
    python inference.py --prompt "Once upon a time" --max-tokens 100

    # Batch processing
    python inference.py --prompts-file prompts.txt --output results.txt

    # Interactive session
    python inference.py --interactive

    # Perplexity evaluation
    python inference.py --perplexity-file test.txt
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast

from cs336_basics.generation.decoding import (
    compute_perplexity,
    decode_text,
    generate_completions,
    interactive_generation,
)
from cs336_basics.nn.models import TransformerLM
from cs336_basics.tokenization.tokenizer import Tokenizer
from cs336_basics.training.checkpoint import load_checkpoint
from cs336_basics.training.optimizers import AdamW


@dataclass
class ModelConfig:
    """
    Configuration for the TinyStories Transformer model.

    This should match the configuration used during training.
    """

    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    eps: float = 1e-5

    @classmethod
    def from_checkpoint_config(cls, config_path: str) -> ModelConfig:
        """Load configuration from the training config file."""
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            return cls(
                vocab_size=config_dict.get("vocab_size", 10000),
                context_length=config_dict.get("context_length", 256),
                d_model=config_dict.get("d_model", 512),
                num_layers=config_dict.get("num_layers", 4),
                num_heads=config_dict.get("num_heads", 16),
                d_ff=config_dict.get("d_ff", 1344),
                rope_theta=config_dict.get("rope_theta", 10000.0),
                eps=config_dict.get("eps", 1e-5),
            )
        except Exception as e:
            warnings.warn(f"Failed to load config from {config_path}: {e}. Using defaults.")
            return cls()


@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""

    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float | None = None
    end_token: str = "<|endoftext|>"
    batch_size: int = 1
    use_amp: bool = True
    torch_compile: bool = True


class TinyStoriesInferenceEngine:
    """
    High-performance inference engine for TinyStories Transformer models.

    This class provides text generation capabilities with support for
    various decoding strategies and batch processing.
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str | None = None,
        merges_path: str | None = None,
        config_path: str | None = None,
        device: str | None = None,
        use_amp: bool = True,
        torch_compile: bool = True,
    ) -> None:
        """
        Initialize the inference engine.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            vocab_path: Path to tokenizer vocabulary file (auto-detected if None)
            merges_path: Path to tokenizer merges file (auto-detected if None)
            config_path: Path to model config file (auto-detected if None)
            device: Target device ('cuda', 'cpu', or None for auto-detection)
            use_amp: Whether to use automatic mixed precision
            torch_compile: Whether to compile the model for optimization
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.torch_compile = torch_compile

        self.device = self._setup_device(device)

        self.model_config = self._load_model_config(config_path)

        self.tokenizer = self._load_tokenizer(vocab_path, merges_path)

        self.model = self._load_model()

        self.generation_stats = {
            "total_generations": 0,
            "total_tokens": 0,
            "total_time": 0.0,
        }

        print(f"✅ Inference engine initialized successfully!")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {self._count_parameters():,}")
        print(f"   AMP enabled: {self.use_amp}")
        print(f"   Torch compile: {self.torch_compile}")

    def _setup_device(self, device: str | None) -> torch.device:
        """Setup and configure the target device."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        device_obj = torch.device(device)

        if device_obj.type == "cuda":
            if not torch.cuda.is_available():
                warnings.warn("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)

            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")

        return device_obj

    def _load_model_config(self, config_path: str | None) -> ModelConfig:
        """Load model configuration."""
        if config_path is None:
            config_path = self.checkpoint_path.parent / "config.json"

        if Path(config_path).exists():
            return ModelConfig.from_checkpoint_config(config_path)
        else:
            print("⚠️  Config file not found, using default TinyStories configuration")
            return ModelConfig()

    def _load_tokenizer(self, vocab_path: str | None, merges_path: str | None) -> Tokenizer:
        """Load the BPE tokenizer."""
        if vocab_path is None:
            potential_paths = [
                Path("output/tinystories_vocab.json"),
                Path("data/tinystories_vocab.json"),
                self.checkpoint_path.parent / "tinystories_vocab.json",
            ]
            for path in potential_paths:
                if path.exists():
                    vocab_path = str(path)
                    break

        if merges_path is None:
            potential_paths = [
                Path("output/tinystories_merges.pkl"),
                Path("data/tinystories_merges.pkl"),
                self.checkpoint_path.parent / "tinystories_merges.pkl",
            ]
            for path in potential_paths:
                if path.exists():
                    merges_path = str(path)
                    break

        if vocab_path is None or merges_path is None:
            raise FileNotFoundError(
                "Could not find tokenizer files. Please provide vocab_path and merges_path, "
                "or ensure they exist in the expected locations."
            )

        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

        print(f"📝 Loaded tokenizer with vocab size: {len(tokenizer.vocab):,}")
        return tokenizer

    def _load_model(self) -> TransformerLM:
        """Load and initialize the model from checkpoint."""
        model = TransformerLM(
            vocab_size=self.model_config.vocab_size,
            context_length=self.model_config.context_length,
            d_model=self.model_config.d_model,
            num_layers=self.model_config.num_layers,
            num_heads=self.model_config.num_heads,
            d_ff=self.model_config.d_ff,
            rope_theta=self.model_config.rope_theta,
            eps=self.model_config.eps,
            device=self.device,
        )

        optimizer = AdamW(model.parameters(), lr=1e-4)

        try:
            iteration = load_checkpoint(self.checkpoint_path, model, optimizer)
            print(f"📦 Loaded checkpoint from step {iteration:,}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        model.eval()

        model = model.to(self.device)

        if self.torch_compile and self.device.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("⚡ Model compiled for execution")
            except Exception as e:
                warnings.warn(f"Model compilation failed: {e}")

        return model

    def _count_parameters(self) -> int:
        """Count the number of model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float | None = None,
        end_token: str = "<|endoftext|>",
    ) -> str:
        """
        Generate text completion for a single prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p nucleus sampling threshold
            end_token: Token that signals end of generation

        Returns:
            Generated text completion
        """
        start_time = time.time()

        with torch.inference_mode():
            if self.use_amp:
                with autocast():
                    completion = decode_text(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        end_token=end_token,
                        device=self.device,
                    )
            else:
                completion = decode_text(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    end_token=end_token,
                    device=self.device,
                )

        generation_time = time.time() - start_time
        generated_tokens = len(self.tokenizer.encode(completion)) - len(self.tokenizer.encode(prompt))

        self.generation_stats["total_generations"] += 1
        self.generation_stats["total_tokens"] += generated_tokens
        self.generation_stats["total_time"] += generation_time

        return completion

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float | None = None,
        end_token: str = "<|endoftext|>",
    ) -> list[str]:
        """
        Generate completions for multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Top-p nucleus sampling threshold
            end_token: Token that signals end of generation

        Returns:
            List of generated completions
        """
        start_time = time.time()

        with torch.inference_mode():
            if self.use_amp:
                with autocast():
                    completions = generate_completions(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompts=prompts,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        end_token=end_token,
                        device=self.device,
                    )
            else:
                completions = generate_completions(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    end_token=end_token,
                    device=self.device,
                )

        generation_time = time.time() - start_time
        total_generated_tokens = sum(
            len(self.tokenizer.encode(completion)) - len(self.tokenizer.encode(prompt))
            for completion, prompt in zip(completions, prompts)
        )

        self.generation_stats["total_generations"] += len(prompts)
        self.generation_stats["total_tokens"] += total_generated_tokens
        self.generation_stats["total_time"] += generation_time

        return completions

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of a text sequence under the model.

        Args:
            text: Input text to evaluate

        Returns:
            Perplexity value
        """
        with torch.inference_mode():
            if self.use_amp:
                with autocast():
                    perplexity = compute_perplexity(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        text=text,
                        device=self.device,
                    )
            else:
                perplexity = compute_perplexity(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    text=text,
                    device=self.device,
                )

        return perplexity

    def interactive_session(self) -> None:
        """Start an interactive text generation session."""
        print("\n🎮 Starting interactive generation session...")
        interactive_generation(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the inference engine."""
        stats = self.generation_stats.copy()

        if stats["total_time"] > 0:
            stats["tokens_per_second"] = stats["total_tokens"] / stats["total_time"]
            stats["avg_generation_time"] = stats["total_time"] / max(stats["total_generations"], 1)
        else:
            stats["tokens_per_second"] = 0.0
            stats["avg_generation_time"] = 0.0

        if self.device.type == "cuda":
            stats["memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        return stats

    def cleanup(self) -> None:
        """Clean up resources and print final statistics."""
        stats = self.get_performance_stats()

        print("\n📊 Inference Statistics:")
        print(f"   Total generations: {stats['total_generations']:,}")
        print(f"   Total tokens generated: {stats['total_tokens']:,}")
        print(f"   Total time: {stats['total_time']:.2f}s")
        print(f"   Tokens per second: {stats['tokens_per_second']:.1f}")
        print(f"   Average generation time: {stats['avg_generation_time']:.3f}s")

        if self.device.type == "cuda":
            print(f"   GPU memory used: {stats.get('memory_allocated_gb', 0):.2f} GB")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def load_prompts_from_file(file_path: str) -> list[str]:
    """Load prompts from a text file (one prompt per line)."""
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_results_to_file(results: list[str], file_path: str) -> None:
    """Save generation results to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result + "\n\n")


def main() -> None:
    """Main entry point for the inference script."""
    parser = argparse.ArgumentParser(
        description="High-Performance Inference for TinyStories Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt generation
  python inference.py --checkpoint checkpoints/checkpoint_final.pt --prompt "Once upon a time"
  
  # Batch processing from file
  python inference.py --checkpoint checkpoints/checkpoint_final.pt --prompts-file prompts.txt
  
  # Interactive session
  python inference.py --checkpoint checkpoints/checkpoint_final.pt --interactive
  
  # Perplexity evaluation
  python inference.py --checkpoint checkpoints/checkpoint_final.pt --perplexity-file test.txt
        """,
    )

    parser.add_argument(
        "--checkpoint", "-c", type=str, default="checkpoints/checkpoint_final.pt", help="Path to model checkpoint file"
    )

    generation_group = parser.add_mutually_exclusive_group()
    generation_group.add_argument("--prompt", "-p", type=str, help="Single prompt for text generation")
    generation_group.add_argument("--prompts-file", type=str, help="File containing prompts (one per line)")
    generation_group.add_argument(
        "--interactive", "-i", action="store_true", help="Start interactive generation session"
    )
    generation_group.add_argument("--perplexity-file", type=str, help="Compute perplexity for text in file")

    parser.add_argument(
        "--vocab-path", type=str, help="Path to tokenizer vocabulary file (auto-detected if not provided)"
    )
    parser.add_argument("--merges-path", type=str, help="Path to tokenizer merges file (auto-detected if not provided)")
    parser.add_argument("--config-path", type=str, help="Path to model config file (auto-detected if not provided)")

    parser.add_argument(
        "--max-tokens", "-m", type=int, default=100, help="Maximum number of new tokens to generate (default: 100)"
    )
    parser.add_argument("--temperature", "-t", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-p", type=float, help="Top-p nucleus sampling threshold")
    parser.add_argument(
        "--end-token", type=str, default="<|endoftext|>", help="End-of-sequence token (default: <|endoftext|>)"
    )

    parser.add_argument("--output", "-o", type=str, help="Output file for batch generation results")

    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu", "auto"], default="auto", help="Target device (default: auto)"
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch compilation")

    args = parser.parse_args()

    if not any([args.prompt, args.prompts_file, args.interactive, args.perplexity_file]):
        parser.error("Must specify one of: --prompt, --prompts-file, --interactive, or --perplexity-file")

    if not Path(args.checkpoint).exists():
        parser.error(f"Checkpoint file not found: {args.checkpoint}")

    try:
        device = None if args.device == "auto" else args.device

        engine = TinyStoriesInferenceEngine(
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab_path,
            merges_path=args.merges_path,
            config_path=args.config_path,
            device=device,
            use_amp=not args.no_amp,
            torch_compile=not args.no_compile,
        )

        if args.interactive:
            engine.interactive_session()

        elif args.prompt:
            print(f"\n🎯 Generating completion for: '{args.prompt}'")
            completion = engine.generate_text(
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                end_token=args.end_token,
            )
            print(f"\n📝 Generated text:\n{completion}")

        elif args.prompts_file:
            print(f"\n📂 Loading prompts from: {args.prompts_file}")
            prompts = load_prompts_from_file(args.prompts_file)
            print(f"   Loaded {len(prompts):,} prompts")

            print("\n🎯 Generating completions...")
            completions = engine.generate_batch(
                prompts=prompts,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                end_token=args.end_token,
            )

            if args.output:
                save_results_to_file(completions, args.output)
                print(f"💾 Results saved to: {args.output}")
            else:
                print("\n📝 Generated completions:")
                for i, completion in enumerate(completions, 1):
                    print(f"\n--- Completion {i} ---")
                    print(completion)

        elif args.perplexity_file:
            print(f"\n📊 Computing perplexity for: {args.perplexity_file}")
            with open(args.perplexity_file, "r", encoding="utf-8") as f:
                text = f.read()

            perplexity = engine.compute_perplexity(text)
            print(f"\n📈 Perplexity: {perplexity:.2f}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    finally:
        if "engine" in locals():
            engine.cleanup()

    return 0


if __name__ == "__main__":
    exit(main())
