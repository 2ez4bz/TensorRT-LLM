"""Parallel input processing using a multiprocessing.Pool.

This module provides a way to offload CPU-bound input processing
(tokenization, multimodal preprocessing) to a pool of worker processes.

Uses `multiprocessing.Pool` (or `torch.multiprocessing.Pool`) under the
hood, but exposes `concurrent.futures.Future` objects so that callers can
use `concurrent.futures.wait` and `concurrent.futures.as_completed`.
"""

import multiprocessing as mp
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple

from ..logger import logger
from .parallel_input_processor import (
    ParallelInputProcessor,
    deserialize_sampling_params,
    serialize_sampling_params,
    set_env_vars,
)

# Module-level globals for worker process state (initialized once per worker).
_worker_tokenizer = None
_worker_input_processor = None
_worker_input_processor_with_hash = None


class MpPoolInputProcessor(ParallelInputProcessor):
    """Manages a pool of worker processes for parallel input processing.

    Usage:
        # Create the processor (typically in LLM.__init__)
        parallel_processor = ParallelInputProcessor(
            num_workers=4,
            model_path="/path/to/model",
            tokenizer_dir=None,  # Uses model_path if None
            checkpoint_format="HF",
            trust_remote_code=True,
            tokenizer_mode="auto",
        )

        # Process inputs (in generate_async)
        future = parallel_processor.process_async(inputs, sampling_params, use_hash=True)
        prompt_token_ids, extra_processed_inputs = future.result()

        # Shutdown when done
        parallel_processor.shutdown()
    """

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
        self.num_workers = num_workers
        self._shutdown = False

        logger.info(f"Starting parallel input processor with {num_workers} workers.")

        # Use "spawn" to shield ourselves from deadlocks that could occur if threads / locks
        # are created in the main process prior to creating the worker processes.
        ctx = mp.get_context("spawn")

        # Pool starts workers eagerly and runs the initializer immediately, so there is no need
        # for a manual readiness check.
        self._pool = ctx.Pool(
            processes=num_workers,
            initializer=_worker_initializer,
            initargs=(
                model_path,
                tokenizer_dir,
                checkpoint_format,
                trust_remote_code,
                tokenizer_mode,
            ),
        )
        logger.info(f"All {num_workers} input processor workers are ready.")

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
        if self._shutdown:
            raise RuntimeError("Processor is shut down")

        # TODO: see to what degree this is necessary.
        sampling_params_dict = serialize_sampling_params(sampling_params)

        future: Future = Future()

        self._pool.apply_async(
            _process_single_input,
            args=(inputs, sampling_params_dict, use_hash),
            callback=future.set_result,
            error_callback=future.set_exception,
        )
        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        if self._shutdown:
            return
        self._shutdown = True

        logger.info("Shutting down parallel input processor.")
        if wait:
            self._pool.close()
            self._pool.join()
        else:
            self._pool.terminate()


def _worker_initializer(
    model_path: str,
    tokenizer_dir: Optional[str],
    checkpoint_format: Optional[str],
    trust_remote_code: bool,
    tokenizer_mode: str,
) -> None:
    """Initialize worker process state.

    Called once per worker when the pool is created. Sets up module-level
    globals that persist for the lifetime of the worker process.
    """
    global _worker_tokenizer, _worker_input_processor, _worker_input_processor_with_hash

    # TODO: see if there's a way to suppress the log spam from these import statements in workers.
    from ..inputs import create_input_processor, create_input_processor_with_hash
    from .llm_utils import ModelLoader

    set_env_vars()

    tokenizer_path = tokenizer_dir or model_path
    _worker_tokenizer = ModelLoader.load_hf_tokenizer(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
        use_fast=(tokenizer_mode != "slow"),
    )
    _worker_input_processor = create_input_processor(
        model_path,
        _worker_tokenizer,
        checkpoint_format,
    )
    _worker_input_processor_with_hash = create_input_processor_with_hash(_worker_input_processor)

    logger.debug("Worker process initialized.")


def _process_single_input(
    inputs: Dict[str, Any],
    sampling_params_dict: Dict[str, Any],
    use_hash: bool,
) -> Tuple[List[int], Optional[Dict[str, Any]]]:
    """Process a single input in a worker process.

    Uses the module-level globals initialized by _worker_initializer.
    """
    sampling_params = deserialize_sampling_params(sampling_params_dict)
    if use_hash and "multi_modal_data" in inputs:
        processor = _worker_input_processor_with_hash
    else:
        processor = _worker_input_processor

    return processor(inputs, sampling_params)
