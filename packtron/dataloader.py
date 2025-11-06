"""
PackTron Dataloader for LLM Training
"""

import os
import logging
import torch
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple

from packtron.utils.dataset_config import PackTronConfig
from packtron.utils import (
    BlendedMegatronDatasetBuilder,
    GPTDataset,
    GPTDatasetConfig,
    get_blend_and_blend_per_split,
    log_single_rank,
)

logger = logging.getLogger(__name__)


class PackTronSampler:
    """
    Megatron-style data sampler for distributed training
    Based on Megatron's MegatronPretrainingSampler
    """
    
    def __init__(self, total_samples, consumed_samples, batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.batch_size = batch_size
        self.data_parallel_rank = data_parallel_rank
        self.batch_times_data_parallel_size = \
            self.batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.batch_size
        end_idx = start_idx + self.batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]

class PackTronDataset:
    def __init__(self, config: PackTronConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.gpt_config = self._create_gpt_dataset_config()
    
    def _create_gpt_dataset_config(self):
        """Create GPTDatasetConfig"""
        blend: Optional[Tuple[List[str], Optional[List[float]]]]
        blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
        # dummy setting
        self.config.data_args_path = None
        self.config.train_data_path= None
        self.config.valid_data_path= None
        self.config.test_data_path= None
        self.config.per_split_data_args_path= None
        blend, blend_per_split = get_blend_and_blend_per_split(self.config)

        # Create GPTDatasetConfig - split_matrix will be calculated automatically in __post_init__
        config = GPTDatasetConfig(
            random_seed=self.config.random_seed,
            sequence_length=self.config.sequence_length,
            blend=blend,  
            blend_per_split=blend_per_split,
            split=self.config.split_config,  # This will be parsed in __post_init__
            num_dataset_builder_threads=1,
            path_to_cache=os.path.join(self.config.path_to_cache, "cache"),
            mmap_bin_files=True,
            tokenizer=self.tokenizer,
            reset_position_ids=True,
            reset_attention_mask=True,
            eod_mask_loss=True,
            create_attention_mask=True
        )
        
        return config
    
    def _build_megatron_datasets(self):
        """
        Build train dataset using BlendedMegatronDatasetBuilder
        exactly like Megatron's train_valid_test_datasets_provider
        """
        log_single_rank(logger, logging.INFO, "Building datasets with BlendedMegatronDatasetBuilder")
        
        # Calculate sample sizes for training
        train_samples = (self.config.train_iters + 10) * self.config.batch_size # ensure enough samples
        valid_samples = (self.config.eval_iters + 10) * self.config.batch_size
        train_val_test_num_samples = [train_samples, valid_samples, 0]
        
        def is_dataset_built_on_rank():
            return True  # Always build on current rank for our use case
        
        # Build datasets exactly like Megatron
        builder = BlendedMegatronDatasetBuilder(
            GPTDataset,
            train_val_test_num_samples,
            is_dataset_built_on_rank,
            self.gpt_config
        )
        
        train_ds, valid_ds = builder.build()
        
        log_single_rank(logger, logging.INFO, f"Built dataset with {len(train_ds) if train_ds else 0} samples")
        
        return train_ds, valid_ds


def build_pretraining_data_loader(dataset, consumed_samples, batch_size,  
                                 data_parallel_rank, data_parallel_size, 
                                 num_workers=0, pin_memory=True, drop_last=True):
    """
    Build dataloader given an input dataset using Megatron-style sampler
    Based on Megatron's build_pretraining_data_loader
    """
    if dataset is None:
        return None
    
    # Create Megatron-style sampler
    batch_sampler = PackTronSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        batch_size=batch_size,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        drop_last=drop_last
    )
    
    # Create DataLoader with Megatron-style sampler (no collate_fn needed)
    # PyTorch automatically handles dictionary batching
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )


def create_dataloader(tokenizer, config, rank: int = 0, 
                               world_size: int = 1, consumed_samples: int = 0) -> DataLoader:
    """
    Factory function to create PackTron DataLoader using Megatron-style data processing
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer instance
        config: PackTronConfig instance
        rank: Current process rank
        world_size: Total number of processes
        consumed_samples: Number of samples already consumed (for checkpoint resuming)
        
    Returns:
        DataLoader instance
    """
    
    # Create dataset wrapper
    loader = PackTronDataset(config, tokenizer)
    
    train_dataset, eval_dataset = loader._build_megatron_datasets()
    
    
    train_dataloader = build_pretraining_data_loader(
        dataset=train_dataset,
        consumed_samples=consumed_samples,
        batch_size=config.batch_size,
        data_parallel_rank=rank,
        data_parallel_size=world_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with Megatron
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop last incomplete batch
    )

    eval_dataloader = build_pretraining_data_loader(
        dataset=eval_dataset,
        consumed_samples=0,
        batch_size=config.batch_size,
        data_parallel_rank=rank,
        data_parallel_size=world_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with Megatron
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop last incomplete batch
    )
    
    log_single_rank(logger, logging.INFO, f"Rank {rank}: Created PackTron DataLoader with batch_size={config.batch_size}, "
                f"dataset_size={len(train_dataset) if hasattr(train_dataset, '__len__') else 'unknown'}")
    
    return train_dataloader, eval_dataloader
