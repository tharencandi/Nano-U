"""
Linear Diagnostic Probes and Centered Kernel Alignment (CKA) Analyzer.

Implements capacity-constrained linear probes for dense semantic segmentation
and representational similarity metrics (Ballan Lab methodology).
"""

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from typing import Tuple, Dict, Any, Optional


def create_pixel_linear_probe(
    input_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int] = (60, 80),
    name: str = "pixel_linear_probe",
) -> keras.Model:
    """
    Constructs a strictly linear diagnostic probe for dense segmentation.

    The probe upsamples the feature map to the target mask resolution via
    bilinear interpolation and applies a 1x1 convolution (linear projection)
    with bias. No intermediate hidden layers or non-linearities are used.

    Args:
        input_shape: (H_l, W_l, C_l) of the intermediate layer activation.
        target_shape: (H, W) of the ground truth segmentation mask.
        name: Probe model name.

    Returns:
        Compiled Keras model emitting unnormalized logits.
    """
    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    
    # Bilinear upsample to target resolution (60, 80)
    if (input_shape[0], input_shape[1]) != target_shape:
        x = layers.Resizing(
            target_shape[0], target_shape[1], interpolation="bilinear", name=f"{name}_resize"
        )(inp)
    else:
        x = inp

    # Strictly linear 1x1 projection (C_l + 1 trainable parameters)
    out = layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=True,
        activation="linear",
        name=f"{name}_linear_conv",
    )(x)

    model = keras.Model(inputs=inp, outputs=out, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


class PixelLinearProbe:
    """Wrapper managing probe training, inference, and metric extraction."""

    def __init__(
        self,
        stage_name: str,
        input_shape: Tuple[int, int, int],
        target_shape: Tuple[int, int] = (60, 80),
    ):
        self.stage_name = stage_name
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.model = create_pixel_linear_probe(input_shape, target_shape, name=f"probe_{stage_name}")

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 16,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """Train the linear probe on frozen representations."""
        val_data = (X_val, Y_val) if X_val is not None and Y_val is not None else None
        history = self.model.fit(
            X_train,
            Y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
        )
        return history.history

    def predict_probs(self, X: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """Emits predicted probabilities in [0, 1]."""
        logits = self.model.predict(X, batch_size=batch_size, verbose=0)
        return 1.0 / (1.0 + np.exp(-logits))


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes linear Centered Kernel Alignment (CKA) between two representation matrices.

    Args:
        X: Activation matrix of shape (N, d1) or (N, H1, W1, C1).
        Y: Activation matrix of shape (N, d2) or (N, H2, W2, C2).

    Returns:
        Scalar similarity in [0, 1].
    """
    # Flatten spatial dimensions into feature vectors
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if Y.ndim > 2:
        Y = Y.reshape(Y.shape[0], -1)

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    # Mean-center across samples
    X_c = X - np.mean(X, axis=0, keepdims=True)
    Y_c = Y - np.mean(Y, axis=0, keepdims=True)

    # Compute NxN Gram matrices (dual formulation: O(N^2 d) instead of O(d^2 N))
    K = np.dot(X_c, X_c.T)
    L = np.dot(Y_c, Y_c.T)

    # HSIC(X, Y) = tr(K L) = sum(K * L)
    hsic_xy = np.sum(K * L)
    hsic_xx = np.sum(K * K)
    hsic_yy = np.sum(L * L)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0

    return float(np.clip(hsic_xy / denom, 0.0, 1.0))
