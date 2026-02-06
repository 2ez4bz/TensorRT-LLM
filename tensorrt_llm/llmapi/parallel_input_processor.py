"""Parallel input processing base class.

This module defines an interface to offload CPU-bound input processing
(tokenization, multimodal preprocessing) to worker processes.
"""

import abc
import dataclasses
import os
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..logger import logger


class ParallelInputProcessor:
    """Base class for parallel input processing."""

    @abc.abstractmethod
    def __init__(
        self,
        num_workers: int,
        model_path: str,
        tokenizer_dir: Optional[str] = None,
        checkpoint_format: Optional[str] = "HF",
        trust_remote_code: bool = False,
        tokenizer_mode: str = "auto",
    ):
        """Initialize the parallel input processor.

        Args:
            num_workers: Number of worker processes in the pool
            model_path: Path to the model (used for config and default tokenizer)
            tokenizer_dir: Optional separate tokenizer directory
            checkpoint_format: Format of the checkpoint ("HF", "mistral", etc.)
            trust_remote_code: Whether to trust remote code when loading
            tokenizer_mode: Tokenizer mode ("auto" or "slow")
        """

    @abc.abstractmethod
    def process_async(
        self,
        inputs: Dict[str, Any],
        sampling_params,
        use_hash: bool = False,
    ) -> Future:
        """Submit an input for asynchronous processing.

        Args:
            inputs: The prompt inputs (TextPrompt dict)
            sampling_params: SamplingParams instance
            use_hash: Whether to use multimodal hashing

        Returns:
            Future that will contain (prompt_token_ids, extra_processed_inputs)
        """

    def process(
        self,
        inputs: Dict[str, Any],
        sampling_params,
        use_hash: bool = False,
    ) -> Tuple[List[int], Optional[Dict[str, Any]]]:
        """Process an input synchronously.

        Convenience method that submits and waits for the result.
        """
        future = self.process_async(inputs, sampling_params, use_hash)
        return future.result()

    @abc.abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the workers underpinning this implementation.

        Args:
            wait: Whether to wait for pending tasks to complete
        """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


def set_env_vars():
    """Set environment variables to avoid CPU over-subscription."""
    # NOTE: somehow, in a prior version of the `ParallelInputProcessor` that managed its own
    # `mp.Process` pool with `mp.Queue` and a background thread for result retrieval, this was not
    # necessary when the `mp_context` was "fork".
    for k in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[k] = "1"
    torch.set_num_threads(1)


# TODO: investigate to what degree this is necessary.
def serialize_sampling_params(sampling_params) -> Dict[str, Any]:
    """Serialize SamplingParams for cross-process transfer.

    Handles non-picklable fields by excluding them or converting them.
    """
    # Get all field values as a dict.
    params_dict = {}
    for field in dataclasses.fields(sampling_params):
        value = getattr(sampling_params, field.name)

        # Skip non-picklable fields.
        if field.name in ("embedding_bias", "logits_processor", "lookahead_config"):
            if value is not None:
                logger.debug(
                    f"Skipping non-picklable field '{field.name}' in parallel input processing"
                )
            continue

        # Handle torch tensors in other fields (convert to list).
        if hasattr(value, "tolist"):
            value = value.tolist()

        params_dict[field.name] = value

    return params_dict


def deserialize_sampling_params(params_dict: Dict[str, Any]):
    """Reconstruct SamplingParams from serialized dict."""
    from ..sampling_params import SamplingParams

    # Filter out any None values for optional fields that were not and remove internal fields that
    # start with an underscore.
    filtered = {k: v for k, v in params_dict.items() if not k.startswith("_")}

    return SamplingParams(**filtered)
