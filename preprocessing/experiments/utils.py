import random
import numpy as np
import torch as th
import numpy as np


def init_seeds(seed: int) -> None:
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.cuda.manual_seed_all(seed)


def sample_uniform_weights(
    num_weights: int, dim_covariates: int, low: float = 0.0, high: float = 1.0
) -> np.ndarray:
    weights = np.zeros(shape=(num_weights, dim_covariates))
    for i in range(num_weights):
        weights[i] = np.random.uniform(low=low, high=high, size=(dim_covariates))
        weights[i] /= np.linalg.norm(weights[i])
    return weights