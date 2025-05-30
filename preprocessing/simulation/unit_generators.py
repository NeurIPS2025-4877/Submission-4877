from argparse import Namespace
from typing import Tuple

import numpy as np

from dataset.TCGA.qm9_tcga_simulation import (get_TCGA_pca_data,
                                           get_TCGA_unit_features)


def generate_TCGA_unit_features(args: Namespace) -> Tuple[dict, dict]:
    tcga_data = get_TCGA_unit_features("./dataset/TCGA/tcga.p")
    tcga_pca_data = get_TCGA_pca_data(n_components=args.dim_pca_unit)
    all_indices = list(range(len(tcga_data)))
    np.random.shuffle(all_indices)

    in_sample_units = {
        "ids": all_indices[: args.num_in_sample_units],
        "features": tcga_data[: args.num_in_sample_units],
        "pca_features": tcga_pca_data[: args.num_in_sample_units],
    }

    if args.full_dataset:
        out_sample_units = {
            "ids": all_indices[args.num_in_sample_units :],
            "features": tcga_data[args.num_in_sample_units :],
            "pca_features": tcga_pca_data[args.num_in_sample_units :],
        }
    else:
        out_sample_units = {
            "ids": all_indices[
                args.num_in_sample_units : args.num_in_sample_units
                + args.num_out_sample_units
            ],
            "features": tcga_data[
                args.num_in_sample_units : args.num_in_sample_units
                + args.num_out_sample_units
            ],
            "pca_features": tcga_pca_data[
                args.num_in_sample_units : args.num_in_sample_units
                + args.num_out_sample_units
            ],
        }
    return in_sample_units, out_sample_units
