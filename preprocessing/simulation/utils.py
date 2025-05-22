import logging
from argparse import Namespace
from typing import Optional, Tuple, Union

from data.dataset import Dataset
from simulation.data_generator import DataGenerator
from simulation.outcome_generators import OutcomeGenerator
from simulation.tcga.data_generator import TCGADataGenerator
from simulation.tcga.outcome_simulator import TCGASimulator
from simulation.treatment_assignment import (RandomTAP,
                                             TreatmentAssignmentPolicy)
from simulation.treatment_generators import generate_id_to_graph_dict_tcga
from simulation.unit_generators import generate_TCGA_unit_features


def get_treatment_assignment_policy(treatment_ids: list, args: Namespace) -> RandomTAP:
    return RandomTAP(treatment_ids=treatment_ids, args=args)


def get_outcome_generator(
    id_to_graph_dict: dict, args: Namespace
) -> Optional[OutcomeGenerator]:

    outcome_generator = TCGASimulator(
        id_to_graph_dict=id_to_graph_dict,
        noise_mean=args.outcome_noise_mean,
        noise_std=args.outcome_noise_std,
        dim_covariates=args.dim_covariates,
    )

    return outcome_generator


def get_data_generator(
    task: str,
    id_to_graph_dict: dict,
    treatment_assignment_policy: TreatmentAssignmentPolicy,
    outcome_generator: OutcomeGenerator,
    in_sample_dataset: Dataset,
    out_sample_dataset: Dataset,
    args: Namespace,
) -> DataGenerator:

    data_generator = TCGADataGenerator(
        id_to_graph_dict=id_to_graph_dict,
        treatment_assignment_policy=treatment_assignment_policy,
        outcome_generator=outcome_generator,
        in_sample_dataset=in_sample_dataset,
        out_sample_dataset=out_sample_dataset,
        args=args,
    )

    return data_generator


def create_dataset_dicts(
    unit_generator,
    treatment_generator,
    args,
) -> Tuple[dict, dict, dict]:
    in_sample_dataset_dict, out_sample_dataset_dict = {}, {}
    logging.info("Generate units...")
    in_sample_dataset_dict["units"], out_sample_dataset_dict["units"] = unit_generator(
        args=args
    )
    logging.info("Generate treatments...")
    id_to_graph_dict = treatment_generator(args=args)
    in_sample_dataset_dict["id_to_graph_dict"] = id_to_graph_dict
    return in_sample_dataset_dict, out_sample_dataset_dict, id_to_graph_dict


def create_dataset(args: Namespace) -> Tuple[Dataset, Dataset, dict]:
    unit_generator = generate_TCGA_unit_features
    treatment_generator = generate_id_to_graph_dict_tcga
    (
        in_sample_dataset_dict,
        out_sample_dataset_dict,
        id_to_graph_dict,
    ) = create_dataset_dicts(
        unit_generator=unit_generator,
        treatment_generator=treatment_generator,
        args=args,
    )
    in_sample_dataset, out_sample_dataset = Dataset(
        data_dict=in_sample_dataset_dict
    ), Dataset(data_dict=out_sample_dataset_dict)
    return in_sample_dataset, out_sample_dataset, id_to_graph_dict
