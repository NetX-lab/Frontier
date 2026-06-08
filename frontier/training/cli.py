"""
Command-line interface for training execution time prediction models.

This module provides a CLI for standalone training of different model structures
(MoE, Attention, Linear Operations, etc.) without running a full simulation.

Model structure categories:
- attn: Attention operations (prefill, decode, KV cache)
- moe: Mixture of Experts operations (gating, shuffling, grouped GEMM)
- linear_op: Linear operations (MLP, LayerNorm, projections, residual add)
"""

import argparse
import sys
from pathlib import Path

from frontier.logger import init_logger
from frontier.training.moe_trainer import MoETrainer, create_moe_trainer_from_model_config
from frontier.training.linear_op_trainer import LinearOpTrainer, create_linear_op_trainer_from_model_config
from frontier.training.attention_trainer import AttentionTrainer, create_attention_trainer_from_model_config
from frontier.types import MeasurementType
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
)
from frontier.moe_routing_runtime import (
    STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH,
    UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH,
)

logger = init_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train execution time prediction models for Frontier simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train MoE models using model configuration
  python -m frontier.training.cli moe \\
    --dataset_path data/profiling/compute/a100/mixtral_8x7b_moe/moe.csv \\
    --output_dir cache \\
    --model_name mixtral_8x7b_moe \\
    --device a100 \\
    --moe_tensor_parallel_size 1 \\
    --expert_parallel_size 1

  # Train Linear Operation models using model configuration
  python -m frontier.training.cli linear_op \\
    --dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/linear_op.csv \\
    --output_dir cache \\
    --model_name meta-llama/Llama-2-7b-hf \\
    --device a100 \\
    --tensor_parallel_size 1

  # Train Attention models (layer models only, without compute models)
  python -m frontier.training.cli attention \\
    --layer_dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv \\
    --output_dir cache \\
    --model_name meta-llama/Llama-2-7b-hf \\
    --device a100 \\
    --tensor_parallel_size 1

  # Train ALL Attention models (including compute-dependent models)
  python -m frontier.training.cli attention \\
    --layer_dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv \\
    --compute_dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/linear_op.csv \\
    --output_dir cache \\
    --model_name meta-llama/Llama-2-7b-hf \\
    --device a100 \\
    --tensor_parallel_size 1
        """
    )
    
    subparsers = parser.add_subparsers(dest="structure", help="Model structure to train")
    
    # MoE subcommand
    moe_parser = subparsers.add_parser("moe", help="Train MoE models")
    
    # Required arguments
    moe_parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the MoE profiling dataset CSV file"
    )
    moe_parser.add_argument(
        "--output_dir",
        type=str,
        default="cache",
        help="Directory to save trained models (default: cache)"
    )
    moe_parser.add_argument(
        "--measurement_type",
        type=str,
        choices=[measurement_type.value for measurement_type in MeasurementType],
        required=True,
        help="Profiling measurement family for the dataset (CUDA_EVENT or KERNEL_ONLY)"
    )
    
    # Configuration mode 1: Use model configuration
    moe_parser.add_argument(
        "--model_name",
        type=str,
        help="Model name (e.g., 'mixtral_8x7b_moe', 'Qwen/Qwen1.5-MoE-A2.7B')"
    )
    moe_parser.add_argument(
        "--device",
        type=str,
        default="a100",
        help="Device SKU (default: a100)"
    )
    
    # Configuration mode 2: Explicit parameters
    moe_parser.add_argument(
        "--num_experts",
        type=int,
        help="Total number of experts in the model"
    )
    moe_parser.add_argument(
        "--router_topk",
        type=int,
        help="Number of experts selected per token"
    )
    moe_parser.add_argument(
        "--hidden_dim",
        type=int,
        help="Model hidden dimension"
    )
    moe_parser.add_argument(
        "--expert_hidden_dim",
        type=int,
        help="Expert FFN hidden dimension"
    )
    
    # Parallelism parameters
    moe_parser.add_argument(
        "--moe_tensor_parallel_size",
        type=int,
        default=1,
        help="MoE tensor parallel size (default: 1)"
    )
    moe_parser.add_argument(
        "--expert_parallel_size",
        type=int,
        default=1,
        help="Expert parallel size (default: 1)"
    )
    moe_parser.add_argument(
        "--routing_runtime_path",
        type=str,
        choices=[
            STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH,
            UNIFORM_MOE_GATING_ROUTING_RUNTIME_PATH,
        ],
        default=STANDARD_MOE_GATING_ROUTING_RUNTIME_PATH,
        help="MoE gating routing runtime path to select from profiling data "
             "(default: standard_fused_topk)"
    )
    moe_parser.add_argument(
        "--gating_runtime_context",
        type=str,
        choices=[
            DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
            PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
        ],
        default=DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
        help="MoE gating runtime context to select from profiling data "
             "(default: standalone_legacy)"
    )
    
    # Predictor configuration
    moe_parser.add_argument(
        "--predictor_type",
        type=str,
        choices=["random_forest", "linear_regression"],
        default="random_forest",
        help="Type of predictor (default: random_forest)"
    )
    
    # Training parameters
    moe_parser.add_argument(
        "--k_fold_cv_splits",
        type=int,
        default=10,
        help="Number of k-fold CV splits (default: 10)"
    )
    moe_parser.add_argument(
        "--num_training_job_threads",
        type=int,
        default=-1,
        help="Number of parallel training threads (default: -1, use all cores)"
    )
    
    # Random Forest parameters (must match RandomForrestExecutionTimePredictorConfig defaults)
    moe_parser.add_argument(
        "--num_estimators",
        type=int,
        nargs="+",
        default=[250, 500, 750],
        help="Number of estimators for Random Forest (default: [250, 500, 750])"
    )
    moe_parser.add_argument(
        "--max_depth",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Max depth for Random Forest (default: [8, 16, 32])"
    )
    moe_parser.add_argument(
        "--min_samples_split",
        type=int,
        nargs="+",
        default=[2, 5, 10],
        help="Min samples split for Random Forest (default: [2, 5, 10])"
    )

    # Linear Operation subcommand
    linear_op_parser = subparsers.add_parser("linear_op", help="Train Linear Operation models (MLP, LayerNorm, etc.)")

    # Required arguments
    linear_op_parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the linear operation profiling dataset CSV file"
    )
    linear_op_parser.add_argument(
        "--output_dir",
        type=str,
        default="cache",
        help="Directory to save trained models (default: cache)"
    )
    linear_op_parser.add_argument(
        "--measurement_type",
        type=str,
        choices=[measurement_type.value for measurement_type in MeasurementType],
        required=True,
        help="Profiling measurement family for the dataset (CUDA_EVENT or KERNEL_ONLY)"
    )

    # Model configuration
    linear_op_parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    linear_op_parser.add_argument(
        "--device",
        type=str,
        default="a100",
        help="Device SKU (default: a100)"
    )

    # Parallelism parameters
    linear_op_parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)"
    )

    # Predictor configuration
    linear_op_parser.add_argument(
        "--predictor_type",
        type=str,
        choices=["random_forest", "linear_regression"],
        default="random_forest",
        help="Type of predictor (default: random_forest)"
    )

    # is_moe parameter
    linear_op_parser.add_argument(
        "--is_moe",
        action="store_true",
        help="Skip training (for MoE models that use expert layers instead of dense MLP)"
    )

    # Training parameters
    linear_op_parser.add_argument(
        "--k_fold_cv_splits",
        type=int,
        default=10,
        help="Number of k-fold CV splits (default: 10)"
    )
    linear_op_parser.add_argument(
        "--num_training_job_threads",
        type=int,
        default=-1,
        help="Number of parallel training threads (default: -1, use all cores)"
    )

    # Random Forest parameters (must match RandomForrestExecutionTimePredictorConfig defaults)
    linear_op_parser.add_argument(
        "--num_estimators",
        type=int,
        nargs="+",
        default=[250, 500, 750],
        help="Number of estimators for Random Forest (default: [250, 500, 750])"
    )
    linear_op_parser.add_argument(
        "--max_depth",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Max depth for Random Forest (default: [8, 16, 32])"
    )
    linear_op_parser.add_argument(
        "--min_samples_split",
        type=int,
        nargs="+",
        default=[2, 5, 10],
        help="Min samples split for Random Forest (default: [2, 5, 10])"
    )

    # Backward compatibility: 'mlp' as alias for 'linear_op'
    mlp_parser = subparsers.add_parser("mlp", help="[DEPRECATED] Alias for 'linear_op' - Train Linear Operation models")

    # Attention subcommand
    attention_parser = subparsers.add_parser("attention", help="Train Attention models")

    # Required arguments
    attention_parser.add_argument(
        "--layer_dataset_path",
        type=str,
        required=True,
        help="Path to the layer profiling dataset CSV file (attention.csv)"
    )
    attention_parser.add_argument(
        "--output_dir",
        type=str,
        default="cache",
        help="Directory to save trained models (default: cache)"
    )
    attention_parser.add_argument(
        "--measurement_type",
        type=str,
        choices=[measurement_type.value for measurement_type in MeasurementType],
        required=True,
        help="Profiling measurement family for the dataset(s) (CUDA_EVENT or KERNEL_ONLY)"
    )

    # Optional compute dataset path
    attention_parser.add_argument(
        "--compute_dataset_path",
        type=str,
        required=False,
        default=None,
        help="Path to the compute profiling dataset CSV file (linear_op.csv). "
             "OPTIONAL: When not provided, compute-dependent models are skipped: "
             "attn_pre_proj, attn_post_proj, attn_rope, input_layernorm, post_attention_layernorm, add"
    )

    # Model configuration
    attention_parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    attention_parser.add_argument(
        "--device",
        type=str,
        default="a100",
        help="Device SKU (default: a100)"
    )

    # Parallelism and configuration parameters
    attention_parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)"
    )
    attention_parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Block size for KV cache (default: 16)"
    )

    # Predictor configuration
    attention_parser.add_argument(
        "--predictor_type",
        type=str,
        choices=["random_forest", "linear_regression"],
        default="random_forest",
        help="Type of predictor (default: random_forest)"
    )

    # Training parameters
    attention_parser.add_argument(
        "--k_fold_cv_splits",
        type=int,
        default=10,
        help="Number of k-fold CV splits (default: 10)"
    )
    attention_parser.add_argument(
        "--num_training_job_threads",
        type=int,
        default=-1,
        help="Number of parallel training threads (default: -1, use all cores)"
    )

    # Random Forest parameters (must match RandomForrestExecutionTimePredictorConfig defaults)
    attention_parser.add_argument(
        "--num_estimators",
        type=int,
        nargs="+",
        default=[250, 500, 750],
        help="Number of estimators for Random Forest (default: [250, 500, 750])"
    )
    attention_parser.add_argument(
        "--max_depth",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Max depth for Random Forest (default: [8, 16, 32])"
    )
    attention_parser.add_argument(
        "--min_samples_split",
        type=int,
        nargs="+",
        default=[2, 5, 10],
        help="Min samples split for Random Forest (default: [2, 5, 10])"
    )

    return parser.parse_args()


def train_moe(args):
    """Train MoE models."""
    logger.info("=" * 80)
    logger.info("MoE Model Training")
    logger.info("=" * 80)
    
    # Validate dataset path
    if not Path(args.dataset_path).exists():
        logger.error(f"Dataset not found: {args.dataset_path}")
        sys.exit(1)
    
    # Prepare training parameters
    training_params = {
        "k_fold_cv_splits": args.k_fold_cv_splits,
        "num_training_job_threads": args.num_training_job_threads,
        "num_estimators": args.num_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
    }
    
    # Create trainer
    if args.model_name:
        # Use model configuration
        logger.info(f"Using model configuration: {args.model_name}")
        trainer = create_moe_trainer_from_model_config(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            device=args.device,
            moe_tensor_parallel_size=args.moe_tensor_parallel_size,
            expert_parallel_size=args.expert_parallel_size,
            predictor_type=args.predictor_type,
            measurement_type=args.measurement_type,
            routing_runtime_path=args.routing_runtime_path,
            gating_runtime_context=args.gating_runtime_context,
            **training_params
        )
    else:
        # Use explicit parameters
        if not all([args.num_experts, args.router_topk, args.hidden_dim, args.expert_hidden_dim]):
            logger.error("When --model_name is not provided, you must specify:")
            logger.error("  --num_experts, --router_topk, --hidden_dim, --expert_hidden_dim")
            sys.exit(1)
        
        logger.info("Using explicit MoE parameters")
        trainer = MoETrainer(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_experts=args.num_experts,
            router_topk=args.router_topk,
            hidden_dim=args.hidden_dim,
            expert_hidden_dim=args.expert_hidden_dim,
            moe_tensor_parallel_size=args.moe_tensor_parallel_size,
            expert_parallel_size=args.expert_parallel_size,
            predictor_type=args.predictor_type,
            measurement_type=args.measurement_type,
            routing_runtime_path=args.routing_runtime_path,
            gating_runtime_context=args.gating_runtime_context,
            **training_params
        )
    
    # Train models
    try:
        models = trainer.train()
        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Trained models: {list(models.keys())}")
        logger.info(f"Models saved to: {args.output_dir}")
        return 0
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def train_linear_op(args):
    """Train Linear Operation models."""
    logger.info("=" * 80)
    logger.info("Linear Operation Model Training")
    logger.info("=" * 80)

    # Validate dataset path
    if not Path(args.dataset_path).exists():
        logger.error(f"Dataset not found: {args.dataset_path}")
        sys.exit(1)

    # Prepare training parameters
    training_params = {
        "k_fold_cv_splits": args.k_fold_cv_splits,
        "num_training_job_threads": args.num_training_job_threads,
        "num_estimators": args.num_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
    }

    # Create trainer using model configuration
    logger.info(f"Using model configuration: {args.model_name}")
    trainer = create_linear_op_trainer_from_model_config(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size,
        predictor_type=args.predictor_type,
        is_moe=getattr(args, 'is_moe', False),
        measurement_type=args.measurement_type,
        **training_params
    )

    # Train models
    try:
        models = trainer.train()
        if models:  # Only log if models were trained (not skipped due to is_moe)
            logger.info("\n" + "=" * 80)
            logger.info("Training completed successfully!")
            logger.info("=" * 80)
            logger.info(f"Trained models: {list(models.keys())}")
            logger.info(f"Models saved to: {args.output_dir}")
        return 0
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


# Backward compatibility alias
def train_mlp(args):
    """[DEPRECATED] Train MLP models - use train_linear_op instead."""
    logger.warning("=" * 80)
    logger.warning("DEPRECATION WARNING: 'mlp' subcommand is deprecated.")
    logger.warning("Please use 'linear_op' subcommand instead.")
    logger.warning("=" * 80)
    return train_linear_op(args)


def train_attention(args):
    """Train Attention models."""
    logger.info("=" * 80)
    logger.info("Attention Model Training")
    logger.info("=" * 80)

    # Validate dataset paths
    # layer_dataset_path is always required
    if not Path(args.layer_dataset_path).exists():
        logger.error(f"Layer dataset not found: {args.layer_dataset_path}")
        sys.exit(1)

    # compute_dataset_path is optional
    compute_dataset_path = args.compute_dataset_path
    if compute_dataset_path is not None:
        if not Path(compute_dataset_path).exists():
            logger.error(f"Compute dataset not found: {compute_dataset_path}")
            sys.exit(1)
        logger.info(f"Compute dataset provided: {compute_dataset_path}")
        logger.info("  -> Will train ALL models (compute + layer + common)")
    else:
        logger.info("Compute dataset NOT provided (--compute_dataset_path not specified)")
        logger.info("  -> Will train LAYER models only (attn_kv_cache_save, attn_prefill, attn_decode, attn_prefill_mixed, attn_decode_in_mixed)")
        logger.info("  -> Skipping compute-dependent models: attn_pre_proj, attn_post_proj, attn_rope, input_layernorm, post_attention_layernorm, add")

    # Prepare training parameters
    training_params = {
        "k_fold_cv_splits": args.k_fold_cv_splits,
        "num_training_job_threads": args.num_training_job_threads,
        "num_estimators": args.num_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
    }

    # Create trainer using model configuration
    logger.info(f"Using model configuration: {args.model_name}")
    trainer = create_attention_trainer_from_model_config(
        layer_dataset_path=args.layer_dataset_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        compute_dataset_path=compute_dataset_path,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        predictor_type=args.predictor_type,
        measurement_type=args.measurement_type,
        **training_params
    )

    # Train models
    try:
        models = trainer.train()
        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Trained models: {list(models.keys())}")
        logger.info(f"Models saved to: {args.output_dir}")
        return 0
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    args = parse_args()

    if args.structure is None:
        logger.error("Please specify a model structure to train (e.g., 'moe', 'linear_op', 'attention')")
        logger.error("Run with --help for usage information")
        sys.exit(1)

    if args.structure == "moe":
        return train_moe(args)
    elif args.structure == "linear_op":
        return train_linear_op(args)
    elif args.structure == "mlp":
        # Backward compatibility: redirect to linear_op
        return train_mlp(args)
    elif args.structure == "attention":
        return train_attention(args)
    else:
        logger.error(f"Unknown structure: {args.structure}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
