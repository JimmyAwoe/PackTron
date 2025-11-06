# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

import logging
from enum import Enum
from typing import List, Optional, Tuple

import numpy

from .msc_utils import open_file
import json

from .helpers_cpp import *
from .helpers_cpp import build_sample_idx_int32, build_sample_idx_int64

logger = logging.getLogger(__name__)


class Split(Enum):
    train = 0
    valid = 1



def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """

    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w


def get_blend_from_list(
    blend: Optional[List[str]],
) -> Optional[Tuple[List[str], Optional[List[float]]]]:
    # pylint: disable=line-too-long
    """Get the blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend
    from the blend list

    Args:
        blend (Optional[List[str]]): The blend list, which can be either
            (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or
            (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        Optional[Tuple[List[str], Optional[List[float]]]]: The blend, consisting of a list of dataset prefixes and optionally a list of dataset weights, e.g. [["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], [30.0, 70.0]].
    """
    # pylint: enable=line-too-long
    if blend is None:
        return None

    if len(blend) % 2 == 1:
        weight_per_dataset = None
        raw_prefix_per_dataset = blend
    else:
        raw_weight_per_dataset, raw_prefix_per_dataset = zip(
            *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
        )

        weight_per_dataset = []
        for rwpd in raw_weight_per_dataset:
            try:
                weight = float(rwpd)
            except ValueError:
                weight = None
            weight_per_dataset.append(weight)

        is_none = map(lambda _: _ is None, weight_per_dataset)
        if any(is_none):
            assert all(is_none)
            weight_per_dataset = None
            raw_prefix_per_dataset = blend

    prefix_per_dataset = [rppd.strip() for rppd in raw_prefix_per_dataset]

    return prefix_per_dataset, weight_per_dataset


def get_blend_and_blend_per_split(args):
    """Get blend and blend_per_split from passed-in arguments."""
    use_data_path = args.data_path is not None or args.data_args_path is not None
    use_per_split_data_path = (
        any(
            elt is not None
            for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]
        )
        or args.per_split_data_args_path is not None
    )

    blend = None
    blend_per_split = None
    if use_data_path:
        if args.data_args_path is not None:
            assert args.data_path is None
            with open_file(args.data_args_path, 'r') as f:
                blend = get_blend_from_list(f.read().split())
        else:
            assert args.data_path is not None
            blend = get_blend_from_list(args.data_path)
    elif use_per_split_data_path:
        if args.per_split_data_args_path is not None:
            with open_file(args.per_split_data_args_path, 'r') as f:
                per_split_data_args = json.load(f)
                # Each element in blend_per_split should be a list of files (and optional
                # weights), so split string if needed.
                for split in ["train", "valid", "test"]:
                    if isinstance(per_split_data_args[split], str):
                        per_split_data_args[split] = per_split_data_args[split].split()

                blend_per_split = [
                    get_blend_from_list(per_split_data_args["train"]),
                    get_blend_from_list(per_split_data_args["valid"]),
                    get_blend_from_list(per_split_data_args["test"]),
                ]
        else:
            blend_per_split = [
                get_blend_from_list(args.train_data_path),
                get_blend_from_list(args.valid_data_path),
                get_blend_from_list(args.test_data_path),
            ]
    else:
        blend, blend_per_split = None, None

    return blend, blend_per_split

def build_sample_idx(
    sizes: numpy.ndarray,
    document_indices: numpy.ndarray,
    sequence_length: int,
    num_epochs: int,
    tokens_per_epoch: int,
    drop_last_partial_sequence: bool = True,
    add_extra_token_to_sequence: bool = True,
):
    """Build the 2-D sample index using the properly typed templated C++ function from helpers.cpp

    Args:
        sizes (numpy.ndarray): The 1-D array of document lengths

        document_indices (numpy.ndarray): The 1-D array of document indices

        sequence_length (int): The sequence length

        num_epochs (int): The number of epochs

        tokens_per_epoch (int): The number of tokens per epoch

        drop_last_partial_sequence (bool): Whether to omit the last partial sequence in the sample
            index should it exist. Defaults to True.

        add_extra_token_to_sequence (bool): Whether to build samples with sequence length
            `sequence_length + 1`. Defaults to True.

    Returns:
        numpy.ndarray: The 2-D sample index
    """

    sample_idx_max = max(document_indices.shape[0], sizes.max())
    if sample_idx_max <= numpy.iinfo(numpy.int32).max:
        sample_idx = build_sample_idx_int32(
            sizes,
            document_indices,
            sequence_length,
            num_epochs,
            tokens_per_epoch,
            drop_last_partial_sequence,
            1 if add_extra_token_to_sequence else 0,
        )
        assert sample_idx.min() >= 0 and sample_idx.max() <= sample_idx_max
    else:
        sample_idx = build_sample_idx_int64(
            sizes,
            document_indices,
            sequence_length,
            num_epochs,
            tokens_per_epoch,
            drop_last_partial_sequence,
            1 if add_extra_token_to_sequence else 0,
        )
    return sample_idx
