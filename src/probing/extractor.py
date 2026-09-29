"""
Layer Activation Extractor for Nano-U.

Extracts intermediate representations across the encoder, bottleneck, and
decoder hierarchy from quantized TFLite (.tflite) and Keras FP32 models.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any


class LayerActivationExtractor:
    """Extracts internal representations from Nano-U at designated architectural depths."""

    DEFAULT_STAGES = [
        "enc1b",
        "enc2b",
        "enc3b",
        "bottleneck_b",
        "dec1_conv_b",
        "dec2_conv_b",
        "dec3_conv_b",
    ]

    STAGE_NAMES = {
        "enc1b": "Encoder 1 (60x80x4)",
        "enc2b": "Encoder 2 (30x40x8)",
        "enc3b": "Encoder 3 (15x20x16)",
        "bottleneck_b": "Bottleneck (5x10x16)",
        "dec1_conv_b": "Decoder 1 (15x20x16)",
        "dec2_conv_b": "Decoder 2 (30x40x8)",
        "dec3_conv_b": "Decoder 3 (60x80x4)",
    }

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.is_tflite = model_path.lower().endswith(".tflite")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        if self.is_tflite:
            self._init_tflite()
        else:
            self._init_keras()

    def _init_tflite(self):
        """Initialize TFLite interpreter with intermediate tensor preservation."""
        self.interpreter = tf.lite.Interpreter(
            model_path=self.model_path,
            experimental_preserve_all_tensors=True,
        )
        self.interpreter.allocate_tensors()

        in_det = self.interpreter.get_input_details()[0]
        out_det = self.interpreter.get_output_details()[0]
        self.input_index = in_det["index"]
        self.output_index = out_det["index"]
        self.input_shape = in_det["shape"]
        self.in_scale, self.in_zero = in_det["quantization"]

        # Map internal tensor names to indices and quantization parameters
        self.tensor_registry: Dict[str, Dict[str, Any]] = {}
        all_details = self.interpreter.get_tensor_details()

        for d in all_details:
            name = d["name"]
            # Look for final activation of each designated block (ReLU after PW conv)
            for stage in self.DEFAULT_STAGES:
                pattern = f"quant_{stage}_relu2"
                if pattern in name:
                    scale, zero = d["quantization"]
                    self.tensor_registry[stage] = {
                        "index": d["index"],
                        "name": name,
                        "shape": d["shape"],
                        "scale": scale,
                        "zero_point": zero,
                        "dtype": d["dtype"],
                    }
                    break

        print(f"[Extractor] Initialized TFLite extractor for {os.path.basename(self.model_path)}")
        print(f"[Extractor] Discovered {len(self.tensor_registry)} intermediate probeable stages:")
        for k, v in self.tensor_registry.items():
            print(f"  - {k:15s} -> shape={v['shape']} scale={v['scale']:.5f}")

    def _init_keras(self):
        """Initialize Keras Functional model."""
        import tf_keras as keras
        self.keras_model = keras.models.load_model(self.model_path, compile=False)
        self.tensor_registry = {}
        for stage in self.DEFAULT_STAGES:
            # find layer matching stage
            for layer in self.keras_model.layers:
                if stage in layer.name and ("relu" in layer.name or "conv" in layer.name):
                    self.tensor_registry[stage] = {
                        "layer": layer,
                        "name": layer.name,
                        "shape": layer.output.shape,
                    }
                    break

    def extract_single_image(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all target stage representations for a single image array (H, W, C)."""
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)

        activations = {}
        if self.is_tflite:
            if self.in_scale > 0:
                inp_q = np.clip(
                    np.round(img / self.in_scale) + self.in_zero, -128, 127
                ).astype(np.int8)
            else:
                inp_q = img.astype(np.float32)

            self.interpreter.set_tensor(self.input_index, inp_q)
            self.interpreter.invoke()

            for stage, meta in self.tensor_registry.items():
                raw = self.interpreter.get_tensor(meta["index"])
                scale = meta["scale"]
                zero = meta["zero_point"]
                if scale > 0:
                    deq = (raw.astype(np.float32) - zero) * scale
                else:
                    deq = raw.astype(np.float32)
                activations[stage] = deq
        return activations

    def extract_dataset(
        self, dataset: tf.data.Dataset, target_stages: Optional[List[str]] = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Extract internal activations, ground truth masks, and raw images for a dataset.

        Returns:
            activations: dict mapping stage_name -> array of shape (N, H_l, W_l, C_l)
            masks: ground truth binary masks of shape (N, H, W, 1)
            images: input images of shape (N, H, W, C)
        """
        stages = target_stages or list(self.tensor_registry.keys())
        stage_lists = {stage: [] for stage in stages}
        mask_list = []
        img_list = []

        for batch_imgs, batch_masks in dataset:
            batch_np = batch_imgs.numpy()
            batch_masks_np = batch_masks.numpy()
            b_size = batch_np.shape[0]

            mask_list.append(batch_masks_np)
            img_list.append(batch_np)

            # Process image by image to support intermediate TFLite tensor inspection
            for i in range(b_size):
                act = self.extract_single_image(batch_np[i : i + 1])
                for s in stages:
                    stage_lists[s].append(act[s])

        activations = {
            s: np.concatenate(stage_lists[s], axis=0) for s in stages
        }
        all_masks = np.concatenate(mask_list, axis=0)
        all_images = np.concatenate(img_list, axis=0)

        return activations, all_masks, all_images
