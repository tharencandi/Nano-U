"""Quantization-Aware Distillation (QAD) pipeline: Teacher → Student → INT8 TFLite.

  Phase 1 — Train Teacher (bu_net)
  Phase 1b — Teacher Quality Gate (val_binary_iou ≥ TEACHER_MIOU_GATE)
  Phase 2 — Train Student (nano_u) via knowledge distillation + QAT from epoch 1
  Phase 3 — Export Student to INT8 TFLite + produce quant_params.json for firmware
  Phase 4 — Train No-Distillation Control (nano_u, pure supervised BCE + QAT)
  Phase 5 — Export Control to INT8 TFLite
  Phase 6 — Evaluate Teacher (float), Student (INT8), and Control (INT8) on test set
             Saves pipeline_summary.json with all metrics

Usage:
    python scripts/run_qad.py
    python scripts/run_qad.py --config config/TinyAgri_config.yaml
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_training, export_int8

# ---------------------------------------------------------------------------
# Teacher quality gate: the minimum val_binary_iou (at the best-val-loss
# epoch) the BU-Net must reach before we trust its representations enough
# to distill from them.  Fold 3 reached 0.8857; a gate of 0.80 leaves
# comfortable margin while catching dead teachers (val_iou ≈ 0.25).
# ---------------------------------------------------------------------------
TEACHER_MIOU_GATE = 0.80


def _read_teacher_best_val_iou(pipeline_dir: str) -> float:
    """Read the teacher's best-epoch val_binary_iou from the saved history."""
    history_path = Path(pipeline_dir) / "history.json"
    if not history_path.exists():
        return 0.0
    with open(history_path) as f:
        history = json.load(f)
    val_losses = history.get("val_loss", [])
    val_ious = history.get("val_binary_iou", [])
    if not val_losses or not val_ious:
        return 0.0
    best_epoch = int(min(range(len(val_losses)), key=lambda i: val_losses[i]))
    return float(val_ious[best_epoch])


def main():
    parser = argparse.ArgumentParser(description="QAD pipeline: Teacher → Student → INT8 TFLite")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config_path = args.config

    from src.utils.config import load_config
    config = load_config(config_path)
    output_dir = config.get("data", {}).get("paths", {}).get("results_dir", "results/")

    print(f"\n{'='*55}")
    print("TEACHER → STUDENT DISTILLATION PIPELINE")
    print(f"{'='*55}")

    # ── Phase 1: Train Teacher ────────────────────────────────────────────
    print("\nPhase 1: Training Teacher")
    teacher_res = run_training("bu_net", config_path, output_dir)

    if teacher_res["status"] != "success":
        print(f"Teacher training failed: {teacher_res.get('error')}")
        sys.exit(1)

    print(f"Teacher trained  →  {teacher_res['model_path']}")

    # ── Phase 1b: Teacher Quality Gate ────────────────────────────────────
    teacher_val_iou = _read_teacher_best_val_iou(teacher_res["pipeline_dir"])
    print(f"\n[GATE] Teacher best-epoch val_binary_iou = {teacher_val_iou:.4f}  "
          f"(threshold ≥ {TEACHER_MIOU_GATE})")

    if teacher_val_iou < TEACHER_MIOU_GATE:
        print(f"\n{'!'*55}")
        print(f"  TEACHER QUALITY GATE FAILED")
        print(f"  val_binary_iou {teacher_val_iou:.4f} < {TEACHER_MIOU_GATE}")
        print(f"  The teacher has not converged — distillation would")
        print(f"  produce a corrupted student. Aborting this fold.")
        print(f"{'!'*55}\n")
        sys.exit(2)

    print(f"[GATE] PASSED — proceeding to student distillation.\n")

    # ── Phase 2: Train Student (QAD) ──────────────────────────────────────
    print("Phase 2: Training Student (QAD)")
    # Inject the teacher path resolved at runtime so the student always distils
    # from the model trained in Phase 1, regardless of what is in the YAML.
    student_res = run_training(
        "nano_u", config_path, output_dir,
        config_overrides={"teacher_weights": teacher_res["model_path"]},
    )

    if student_res["status"] != "success":
        print(f"Student training failed: {student_res.get('error')}")
        sys.exit(1)

    print(f"Student trained  →  {student_res['model_path']}")

    # ── Phase 3: Quantize Student to INT8 TFLite ──────────────────────────
    print("\nPhase 3: Quantize Student to INT8 TFLite")
    student_src = Path(student_res["model_path"])
    models_dir_path = str(student_src.parent)

    if not student_src.exists():
        print(f"Student model not found at {student_src}")
        sys.exit(1)

    quant_res = export_int8(str(student_src), models_dir=models_dir_path,
                            config_path=config_path)

    if quant_res.get("quant_params_path"):
        print(f"\n  ⚠  Firmware dependency: {quant_res['quant_params_path']}")
        print(f"     firmware/build.rs reads this file at compile time.")
        print(f"     Run 'cargo build' only after this file exists.\n")

    if quant_res.get("status") != "success":
        print("Quantization failed — aborting.")
        sys.exit(1)

    # ── Phase 4: Train No-Distillation Control (pure supervised + QAT) ───
    print(f"\n{'='*55}")
    print("Phase 4: No-Distillation Control (pure BCE + QAT)")
    print(f"{'='*55}")

    control_output_dir = str(Path(output_dir) / "nano_u_control")
    control_models_dir = str(Path(models_dir_path))

    control_res = run_training(
        "nano_u", config_path, control_output_dir,
        config_overrides={
            "use_distillation": False,
            "model_name": "nano_u_control",
            "models_dir": control_models_dir,
        },
    )

    control_quant_res = {}
    if control_res["status"] == "success":
        print(f"Control trained  →  {control_res['model_path']}")

        # Phase 5: Quantize Control to INT8 TFLite
        control_h5 = Path(control_res["model_path"])
        print("\nPhase 5: Quantize Control to INT8 TFLite")
        if control_h5.exists():
            control_quant_res = export_int8(
                str(control_h5), models_dir=control_models_dir,
                config_path=config_path,
            )
        else:
            print("  Control .h5 not found — skipping quantization.")
    else:
        print(f"Control training failed: {control_res.get('error')}")

    # ── Phase 6: Evaluate on Test Set ─────────────────────────────────────
    print(f"\n{'='*55}")
    print("Phase 6: Evaluation and Visualization")
    print(f"{'='*55}")
    eval_results_all = {}
    try:
        from src.evaluate import evaluate_and_plot

        # Evaluate Teacher (float Keras — used only as distillation reference)
        teacher_path = Path(teacher_res["model_path"])
        teacher_eval_out = Path(teacher_res["pipeline_dir"]) / "eval_predictions.png"
        print(f"\nEvaluating Teacher ({teacher_path.stem})...")
        eval_results_all['teacher'] = evaluate_and_plot(
            model_name=teacher_path.stem,
            config_path=config_path,
            out_path=str(teacher_eval_out),
        )

        # Evaluate Student in BOTH formats so the quantization gap is real:
        #   - FP32 Keras (.h5)  → prefer="float"
        #   - INT8 TFLite       → prefer="tflite" (the on-device format)
        # Without forcing prefer="float" the resolver picks .tflite for both,
        # making the "float" row silently identical to INT8.
        student_dir = Path(student_res["pipeline_dir"])
        print(f"\nEvaluating Student FP32 Keras ({student_src.stem})...")
        eval_results_all['student_fp32'] = evaluate_and_plot(
            model_name=student_src.stem,
            config_path=config_path,
            out_path=str(student_dir / "eval_predictions_fp32.png"),
            prefer="float",
        )

        print(f"\nEvaluating Student INT8 TFLite ({student_src.stem})...")
        eval_results_all['student_int8'] = evaluate_and_plot(
            model_name=student_src.stem,
            config_path=config_path,
            out_path=str(student_dir / "eval_predictions_int8.png"),
            prefer="tflite",
        )

        # Evaluate Control (INT8 TFLite) if available
        control_tflite = Path(control_models_dir) / "nano_u_control.tflite"
        if control_tflite.exists():
            control_eval_dir = Path(control_output_dir)
            control_eval_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nEvaluating Control INT8 TFLite (nano_u_control)...")
            eval_results_all['control_int8'] = evaluate_and_plot(
                model_name="nano_u_control",
                config_path=config_path,
                out_path=str(control_eval_dir / "eval_predictions_int8.png"),
                prefer="tflite",
            )

    except Exception as e:
        print(f"Evaluation failed: {e}")

    # Pipeline summary JSON
    summary = {
        "config_path": config_path,
        "teacher": {
            "model_path": teacher_res["model_path"],
            "pipeline_dir": teacher_res["pipeline_dir"],
            "best_val_iou": teacher_val_iou,
            "gate_threshold": TEACHER_MIOU_GATE,
            "eval": eval_results_all.get("teacher", {}),
        },
        "student": {
            "model_path": student_res["model_path"],
            "pipeline_dir": student_res["pipeline_dir"],
            "tflite_path": quant_res.get("tflite_path"),
            "tflite_size_kb": quant_res.get("size_kb"),
            "quant_params_path": quant_res.get("quant_params_path"),
            "eval_fp32": eval_results_all.get("student_fp32", {}),
            "eval_int8": eval_results_all.get("student_int8", {}),
        },
        "control": {
            "model_path": control_res.get("model_path", "N/A"),
            "pipeline_dir": control_res.get("pipeline_dir", "N/A"),
            "tflite_path": control_quant_res.get("tflite_path", "N/A"),
            "eval_int8": eval_results_all.get("control_int8", {}),
        },
    }
    summary_path = Path(output_dir) / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    teacher_miou = eval_results_all.get("teacher", {}).get("miou", float("nan"))
    student_fp32_miou = eval_results_all.get("student_fp32", {}).get("miou", float("nan"))
    student_miou = eval_results_all.get("student_int8", {}).get("miou", float("nan"))
    control_miou = eval_results_all.get("control_int8", {}).get("miou", float("nan"))

    print(f"\n{'='*55}")
    print("PIPELINE COMPLETE")
    print(f"   Teacher path:     {teacher_res['model_path']}")
    print(f"   Student path:     {student_res['model_path']}")
    print(f"   TFLite:           {quant_res.get('tflite_path', 'N/A')}")
    print(f"   Size:             {quant_res.get('size_kb', 'N/A')} KB")
    print(f"   Quant params:     {quant_res.get('quant_params_path', 'N/A')}  ← required by firmware/build.rs")
    print(f"   Teacher mIoU:     {teacher_miou:.4f}  (gate: {teacher_val_iou:.4f} ≥ {TEACHER_MIOU_GATE})")
    print(f"   Student mIoU:     {student_fp32_miou:.4f}  (FP32 Keras)")
    print(f"   Student mIoU:     {student_miou:.4f}  (INT8 TFLite / QAD)")
    print(f"   Control mIoU:     {control_miou:.4f}  (INT8 TFLite / no KD)")
    print(f"   Summary JSON:     {summary_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
