from typing import Literal

import numpy as np


VERTEX_GPU_FACTOR = 1e-11


def estimate_duration(
    num_rows: int,
    num_features: int,
    task: Literal["classification", "regression"],
    tabpfn_config: dict = {},
    duration_factor: float = VERTEX_GPU_FACTOR,
    latency_offset: float = 0.0,
) -> float:
    """
    Estimates the duration of a prediction task.
    """

    # Logic comes from _estimate_model_usage in base.py of the TabPFN codebase.
    CONSTANT_COMPUTE_OVERHEAD = 8000
    NUM_SAMPLES_FACTOR = 4
    NUM_SAMPLES_PLUS_FEATURES = 6.5
    CELLS_FACTOR = 0.25
    CELLS_SQUARED_FACTOR = 1.3e-7

    EMBEDDING_SIZE = 192
    NUM_HEADS = 6
    NUM_LAYERS = 12
    FEATURES_PER_GROUP = 2

    n_estimators = tabpfn_config.get(
        "n_estimators", 4 if task == "classification" else 8
    )

    num_samples = num_rows
    num_feature_groups = int(np.ceil(num_features / FEATURES_PER_GROUP))

    num_cells = (num_feature_groups + 1) * num_samples
    compute_cost = (EMBEDDING_SIZE**2) * NUM_HEADS * NUM_LAYERS

    base_duration = (
        n_estimators
        * compute_cost
        * (
            CONSTANT_COMPUTE_OVERHEAD
            + num_samples * NUM_SAMPLES_FACTOR
            + (num_samples + num_feature_groups) * NUM_SAMPLES_PLUS_FEATURES
            + num_cells * CELLS_FACTOR
            + num_cells**2 * CELLS_SQUARED_FACTOR
        )
    )

    return round(base_duration * duration_factor + latency_offset, 3)
