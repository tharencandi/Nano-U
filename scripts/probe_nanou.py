#!/usr/bin/env python3
"""
CLI Tool: Representation Probing on Nano-U (Stage 2 Preliminaries).

Trains linear diagnostic probes across Nano-U layer hierarchy,
evaluates linear decodability of traversable terrain, computes
Centered Kernel Alignment (CKA), and exports publication-ready figures.
"""

import os
import sys
import argparse

# Ensure Nano-U project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.probing.evaluator import run_probing_suite


def main():
    parser = argparse.ArgumentParser(
        description="Run layer-wise representation probing on Nano-U (Ballan crossover)."
    )
    parser.add_argument(
        "--model",
        default="models/TinyAgri/nano_u.tflite",
        help="Path to .tflite or .keras model file (default: models/TinyAgri/nano_u.tflite)",
    )
    parser.add_argument(
        "--config",
        default="config/TinyAgri_config.yaml",
        help="Path to dataset YAML config (default: config/TinyAgri_config.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train each linear probe (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for probe training and inference (default: 16)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for probing reports and figures (default: results/probing)",
    )
    parser.add_argument(
        "--no-export-thesis",
        action="store_true",
        help="Disable automatic export of figures to thesis-manuscript/images/probing/",
    )

    args = parser.parse_args()

    # Resolve relative paths with respect to PROJECT_ROOT
    model_path = args.model if os.path.isabs(args.model) else os.path.join(PROJECT_ROOT, args.model)
    config_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)

    run_probing_suite(
        model_path=model_path,
        config_path=config_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        results_dir=args.output,
        export_to_thesis=not args.no_export_thesis,
    )


if __name__ == "__main__":
    main()
