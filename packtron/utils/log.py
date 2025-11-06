# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
import logging 
from typing import Any

def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any):
    """If torch distributed is initialized, write log on only one rank

    Args:
        logger (logging.Logger): The logger to write the logs

        args (Tuple[Any]): All logging.Logger.log positional arguments

        rank (int, optional): The rank to write on. Defaults to 0.

        kwargs (Dict[str, Any]): All logging.Logger.log keyword arguments
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank:
            logger.log(*args, **kwargs)
    else:
        logger.log(*args, **kwargs)