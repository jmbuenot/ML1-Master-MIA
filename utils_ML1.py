"""Utility functions collected from Units 1 and 2 analyses.

This module ports the Julia-style helper routines documented in the
course notes to Python/Numpy implementations so they can be reused in
experiments or notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------------
# Basic helper functions from the Julia notes
# ---------------------------------------------------------------------------

def add(x: float, y: float) -> float:
    """Return the sum of *x* and *y*.

    Mirrors the simple one-line Julia definition provided in the Unit 2
    notes and is mainly illustrative.
    """

    return float(x + y)


def mse(outputs: ArrayLike, targets: ArrayLike) -> float:
    """Compute the mean squared error between *outputs* and *targets*."""

    outputs_arr = np.asarray(outputs, dtype=float)
    targets_arr = np.asarray(targets, dtype=float)
    diff = targets_arr - outputs_arr
    return float(np.mean(np.square(diff)))


def avg_greater_than_zero(values: ArrayLike) -> float:
    """Average only the positive entries of *values*.

    This corresponds to the multi-line Julia example that first builds a
    boolean mask and then evaluates the mean.
    """

    values_arr = np.asarray(values, dtype=float)
    positives = values_arr > 0
    if not np.any(positives):
        raise ValueError("No positive values available to compute the average.")
    return float(np.mean(values_arr[positives]))


def avg_greater_than_zero_compact(values: ArrayLike) -> float:
    """Compact variant that mirrors the one-line Julia definition."""

    values_arr = np.asarray(values, dtype=float)
    positives = values_arr > 0
    if not np.any(positives):
        raise ValueError("No positive values available to compute the average.")
    return float(np.mean(values_arr[positives]))


def avg_greater_than_zero_with_mask(values: ArrayLike) -> Tuple[np.ndarray, float]:
    """Return the boolean mask of positive entries and their average.

    This reproduces the tuple-returning function shown in the Julia
    notes, returning both the mask and the aggregated value.
    """

    values_arr = np.asarray(values, dtype=float)
    positives = values_arr > 0
    if not np.any(positives):
        raise ValueError("No positive values available to compute the average.")
    return positives, float(np.mean(values_arr[positives]))


# ---------------------------------------------------------------------------
# One-hot encoding utilities
# ---------------------------------------------------------------------------

def _ensure_1d(array: Union[Sequence, np.ndarray], name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 1:
        raise ValueError(f"{name} debe ser un vector unidimensional.")
    return arr


def _unique_preserve_order(values: np.ndarray) -> np.ndarray:
    seen = set()
    ordered = []
    for item in values:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return np.array(ordered, dtype=values.dtype)


def one_hot_encoding(
    feature: Union[Sequence, np.ndarray],
    classes: Optional[Union[Sequence, np.ndarray]] = None,
) -> np.ndarray:
    """Vectorise a categorical attribute into boolean columns.

    Parameters
    ----------
    feature:
        Vector con los valores categóricos para cada patrón.
    classes:
        Lista explícita de categorías. Si no se proporciona, se deducen
        preservando el orden de aparición.
    """

    feature_arr = _ensure_1d(feature, "feature")

    if feature_arr.dtype == bool and classes is None:
        return feature_arr.reshape(-1, 1)

    if classes is None:
        classes_arr = _unique_preserve_order(feature_arr)
    else:
        classes_arr = _ensure_1d(classes, "classes")

    if classes_arr.size < 2:
        raise ValueError("Se necesitan al menos dos clases para codificar.")

    if not np.all(np.isin(feature_arr, classes_arr)):
        raise ValueError("'feature' contiene valores que no aparecen en 'classes'.")

    if classes_arr.size == 2:
        return (feature_arr == classes_arr[0]).reshape(-1, 1)

    encoded = feature_arr[:, None] == classes_arr[None, :]
    return encoded.astype(bool)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _ensure_2d_real(matrix: Union[np.ndarray, Sequence[Sequence[float]]], name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} debe ser una matriz bidimensional.")
    return arr


def calculate_min_max_normalization_parameters(
    dataset: Union[np.ndarray, Sequence[Sequence[float]]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-column min and max in row-vector form."""

    data = _ensure_2d_real(dataset, "dataset")
    min_vals = np.min(data, axis=0, keepdims=True)
    max_vals = np.max(data, axis=0, keepdims=True)
    return min_vals, max_vals


def calculate_zero_mean_normalization_parameters(
    dataset: Union[np.ndarray, Sequence[Sequence[float]]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-column mean and standard deviation."""

    data = _ensure_2d_real(dataset, "dataset")
    means = np.mean(data, axis=0, keepdims=True)
    stds = np.std(data, axis=0, ddof=0, keepdims=True)
    return means, stds


def normalize_min_max_inplace(
    dataset: Union[np.ndarray, Sequence[Sequence[float]]],
    normalization_parameters: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """In-place min-max normalisation, matching the Julia helper."""

    data = _ensure_2d_real(dataset, "dataset")
    min_vals, max_vals = normalization_parameters
    min_vals = np.asarray(min_vals, dtype=float)
    max_vals = np.asarray(max_vals, dtype=float)
    data -= min_vals
    scale = max_vals - min_vals
    with np.errstate(divide="ignore", invalid="ignore"):
        data /= scale
    constant_mask = np.isclose(scale, 0.0)
    if np.any(constant_mask):
        data[:, constant_mask.flatten()] = 0.0
    return data


def normalize_min_max(
    dataset: Union[np.ndarray, Sequence[Sequence[float]]],
    normalization_parameters: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Return a min-max normalised copy of *dataset*."""

    data = _ensure_2d_real(dataset, "dataset")
    params = (
        calculate_min_max_normalization_parameters(data)
        if normalization_parameters is None
        else normalization_parameters
    )
    return normalize_min_max_inplace(np.copy(data), params)


def normalize_zero_mean_inplace(
    dataset: Union[np.ndarray, Sequence[Sequence[float]]],
    normalization_parameters: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """In-place zero-mean normalisation (standardisation)."""

    data = _ensure_2d_real(dataset, "dataset")
    means, stds = normalization_parameters
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    data -= means
    with np.errstate(divide="ignore", invalid="ignore"):
        data /= stds
    constant_mask = np.isclose(stds, 0.0)
    if np.any(constant_mask):
        data[:, constant_mask.flatten()] = 0.0
    return data


def normalize_zero_mean(
    dataset: Union[np.ndarray, Sequence[Sequence[float]]],
    normalization_parameters: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Return a zero-mean normalised copy of *dataset*."""

    data = _ensure_2d_real(dataset, "dataset")
    params = (
        calculate_zero_mean_normalization_parameters(data)
        if normalization_parameters is None
        else normalization_parameters
    )
    return normalize_zero_mean_inplace(np.copy(data), params)


# ---------------------------------------------------------------------------
# Classification utilities
# ---------------------------------------------------------------------------

def classify_outputs(outputs: Union[np.ndarray, Sequence[Sequence[float]]], threshold: float = 0.5) -> np.ndarray:
    """Convert model outputs into boolean class assignments.

    Parameters
    ----------
    outputs:
        Matriz con un patrón por fila.
    threshold:
        Umbral para el caso binario.
    """

    outputs_arr = np.asarray(outputs, dtype=float)
    if outputs_arr.ndim != 2:
        raise ValueError("'outputs' debe ser una matriz de dos dimensiones.")

    num_outputs = outputs_arr.shape[1]
    if num_outputs < 1:
        raise ValueError("La matriz de salida debe tener al menos una columna.")

    if num_outputs == 1:
        return (outputs_arr >= threshold).astype(bool)

    max_indices = np.argmax(outputs_arr, axis=1)
    classified = np.zeros_like(outputs_arr, dtype=bool)
    classified[np.arange(outputs_arr.shape[0]), max_indices] = True
    return classified


def accuracy(
    outputs: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
    targets: Union[np.ndarray, Sequence[Sequence[bool]], Sequence[bool]],
    *,
    threshold: float = 0.5,
) -> float:
    """Compute classification accuracy across binary or multiclass setups."""

    outputs_arr = np.asarray(outputs)
    targets_arr = np.asarray(targets)

    if outputs_arr.shape != targets_arr.shape:
        if outputs_arr.ndim == 1 and targets_arr.ndim == 1:
            if targets_arr.dtype == bool:
                return accuracy(outputs_arr.reshape(-1, 1), targets_arr.reshape(-1, 1), threshold=threshold)
        if outputs_arr.ndim == 2 and targets_arr.ndim == 2 and targets_arr.shape[1] == 1:
            return accuracy(outputs_arr[:, 0], targets_arr[:, 0], threshold=threshold)
        if outputs_arr.ndim == 2 and targets_arr.ndim == 1 and targets_arr.dtype == bool:
            return accuracy(outputs_arr, targets_arr.reshape(-1, 1), threshold=threshold)
        raise ValueError("Las dimensiones de 'outputs' y 'targets' no coinciden.")

    if outputs_arr.ndim == 1:
        outputs_bool = outputs_arr >= threshold if outputs_arr.dtype != bool else outputs_arr
        return float(np.mean(outputs_bool == targets_arr))

    if outputs_arr.dtype != bool:
        classified = classify_outputs(outputs_arr, threshold=threshold)
        return accuracy(classified, targets_arr, threshold=threshold)

    if outputs_arr.shape[1] == 1:
        return float(np.mean(outputs_arr[:, 0] == targets_arr[:, 0]))

    comparison = np.all(outputs_arr == targets_arr, axis=1)
    return float(np.mean(comparison))


# ---------------------------------------------------------------------------
# Simple MLP classifier (numpy implementation)
# ---------------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_activation(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


@dataclass
class ActivationFunction:
    func: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray], np.ndarray]


def _parse_activation(activation: Union[str, ActivationFunction, Callable[[np.ndarray], np.ndarray]]) -> ActivationFunction:
    if isinstance(activation, ActivationFunction):
        return activation
    if isinstance(activation, str):
        key = activation.lower()
        if key == "sigmoid":
            return ActivationFunction(sigmoid, sigmoid_derivative)
        if key == "tanh":
            return ActivationFunction(tanh_activation, tanh_derivative)
        if key == "relu":
            return ActivationFunction(relu, relu_derivative)
        raise ValueError(f"Función de activación desconocida: {activation}")
    if callable(activation):
        if hasattr(activation, "derivative") and callable(getattr(activation, "derivative")):
            return ActivationFunction(activation, getattr(activation, "derivative"))
        raise ValueError("Las funciones personalizadas deben exponer un atributo 'derivative'.")
    raise TypeError("Tipo de activación no soportado.")


class MLPClassifierModel:
    """Pequeño perceptrón multicapa entrenable mediante descenso de gradiente."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        hidden_activations: Sequence[ActivationFunction],
        output_activation: str,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Se necesitan al menos capas de entrada y salida.")
        if len(hidden_activations) != len(layer_sizes) - 2:
            raise ValueError("Número de activaciones ocultas incompatible con la topología.")

        self.layer_sizes = list(layer_sizes)
        self.hidden_activations = list(hidden_activations)
        self.output_activation = output_activation
        self.rng = rng or np.random.default_rng()
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (in_size + out_size))
            self.weights.append(self.rng.uniform(-limit, limit, size=(in_size, out_size)))
            self.biases.append(np.zeros(out_size, dtype=float))

    def forward(self, inputs: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [np.asarray(inputs, dtype=float)]
        zs: List[np.ndarray] = []

        for idx, activation in enumerate(self.hidden_activations):
            z = activations[-1] @ self.weights[idx] + self.biases[idx]
            zs.append(z)
            activations.append(activation.func(z))

        z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        zs.append(z_out)
        if self.output_activation == "sigmoid":
            activations.append(sigmoid(z_out))
        elif self.output_activation == "softmax":
            activations.append(softmax(z_out))
        else:
            activations.append(z_out)
        return zs, activations

    def predict_proba(self, inputs: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        _, activations = self.forward(np.asarray(inputs, dtype=float))
        return activations[-1]

    def predict(self, inputs: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        probs = self.predict_proba(inputs)
        return classify_outputs(probs)


def build_class_ann(
    num_inputs: int,
    topology: Sequence[int],
    num_outputs: int,
    *,
    transfer_functions: Optional[Sequence[Union[str, ActivationFunction, Callable[[np.ndarray], np.ndarray]]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> MLPClassifierModel:
    """Crear un perceptrón multicapa para clasificación."""

    topology = list(topology)
    if transfer_functions is None:
        transfer_functions = ["sigmoid"] * len(topology)
    if len(transfer_functions) != len(topology):
        raise ValueError("transfer_functions debe tener tantas entradas como capas ocultas.")

    hidden_activations = [_parse_activation(fn) for fn in transfer_functions]
    layer_sizes = [num_inputs, *topology, num_outputs]
    output_activation = "sigmoid" if num_outputs == 1 else "softmax"
    return MLPClassifierModel(layer_sizes, hidden_activations, output_activation, rng=rng)


def _compute_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    preds = np.clip(predictions, 1e-12, 1.0 - 1e-12)
    if targets.shape[1] == 1:
        loss = -np.mean(targets * np.log(preds) + (1.0 - targets) * np.log(1.0 - preds))
    else:
        loss = -np.mean(np.sum(targets * np.log(preds), axis=1))
    return float(loss)


def train_class_ann(
    topology: Sequence[int],
    dataset: Tuple[Union[np.ndarray, Sequence[Sequence[float]]], Union[np.ndarray, Sequence[Sequence[bool]]]],
    *,
    transfer_functions: Optional[Sequence[Union[str, ActivationFunction, Callable[[np.ndarray], np.ndarray]]]] = None,
    max_epochs: int = 1000,
    min_loss: float = 0.0,
    learning_rate: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[MLPClassifierModel, List[float]]:
    """Entrena una red de clasificación siguiendo los criterios de parada dados."""

    inputs_raw, targets_raw = dataset
    inputs = _ensure_2d_real(inputs_raw, "inputs")

    targets_arr = np.asarray(targets_raw)
    if targets_arr.ndim == 1:
        targets_arr = targets_arr.reshape(-1, 1)
    if targets_arr.dtype != bool:
        targets_arr = targets_arr.astype(bool)

    if inputs.shape[0] != targets_arr.shape[0]:
        raise ValueError("inputs y targets deben tener el mismo número de patrones.")

    ann = build_class_ann(
        num_inputs=inputs.shape[1],
        topology=topology,
        num_outputs=targets_arr.shape[1],
        transfer_functions=transfer_functions,
        rng=rng,
    )

    targets_float = targets_arr.astype(float)
    losses: List[float] = []
    predictions = ann.predict_proba(inputs)
    current_loss = _compute_loss(predictions, targets_float)
    losses.append(current_loss)

    epoch = 0
    while epoch < max_epochs and current_loss > min_loss:
        zs, activations = ann.forward(inputs)
        y_pred = activations[-1]
        delta = (y_pred - targets_float) / inputs.shape[0]

        grads_w: List[np.ndarray] = [np.zeros_like(w) for w in ann.weights]
        grads_b: List[np.ndarray] = [np.zeros_like(b) for b in ann.biases]

        grads_w[-1] = activations[-2].T @ delta
        grads_b[-1] = delta.sum(axis=0)

        back_grad = delta
        for layer_idx in range(len(ann.hidden_activations) - 1, -1, -1):
            back_grad = (back_grad @ ann.weights[layer_idx + 1].T) * ann.hidden_activations[layer_idx].derivative(zs[layer_idx])
            grads_w[layer_idx] = activations[layer_idx].T @ back_grad
            grads_b[layer_idx] = back_grad.sum(axis=0)

        for idx in range(len(ann.weights)):
            ann.weights[idx] -= learning_rate * grads_w[idx]
            ann.biases[idx] -= learning_rate * grads_b[idx]

        epoch += 1
        predictions = ann.predict_proba(inputs)
        current_loss = _compute_loss(predictions, targets_float)
        losses.append(current_loss)

    return ann, losses


def train_class_ann_with_vector_targets(
    topology: Sequence[int],
    dataset: Tuple[Union[np.ndarray, Sequence[Sequence[float]]], Union[np.ndarray, Sequence[bool]]],
    **kwargs,
) -> Tuple[MLPClassifierModel, List[float]]:
    """Conveniencia para manejar etiquetas binarias en forma de vector."""

    inputs, targets_vector = dataset
    targets_matrix = np.reshape(np.asarray(targets_vector, dtype=bool), (-1, 1))
    return train_class_ann(topology, (inputs, targets_matrix), **kwargs)


__all__ = [
    "MLPClassifierModel",
    "ActivationFunction",
    "accuracy",
    "add",
    "avg_greater_than_zero",
    "avg_greater_than_zero_compact",
    "avg_greater_than_zero_with_mask",
    "build_class_ann",
    "calculate_min_max_normalization_parameters",
    "calculate_zero_mean_normalization_parameters",
    "classify_outputs",
    "mse",
    "normalize_min_max",
    "normalize_min_max_inplace",
    "normalize_zero_mean",
    "normalize_zero_mean_inplace",
    "one_hot_encoding",
    "sigmoid",
    "train_class_ann",
    "train_class_ann_with_vector_targets",
]