from abc import ABC

import numpy as np


class OutcomeGenerator(ABC):
    def __init__(
        self, id_to_graph_dict: dict, noise_mean: float = 0.0, noise_std: float = 1.0
    ):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.id_to_graph_dict = id_to_graph_dict

    def _sample_noise(self) -> float:
        return np.random.normal(loc=self.noise_mean, scale=self.noise_std)

def generate_outcome_tcga(
    unit_features: np.ndarray,
    pca_features: np.ndarray,
    prop: np.ndarray,
    random_weights: np.ndarray,
) -> float:
    baseline_effect = 10.0 * np.dot(random_weights[0], unit_features)
    treatment_effect = np.dot(prop[:8], pca_features) * 0.01
    return baseline_effect + treatment_effect
