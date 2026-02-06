"""Parallel input processing using multiprocessing.Process and Queues.

This module provides a way to offload CPU-bound input processing (tokenization,
multimodal preprocessing) to a self-managed pool of workers.
"""

# import multiprocessing as mp
import threading
import uuid
from concurrent.futures import Future
from typing import Any, Dict, List, Optional

import torch.multiprocessing as mp

from ..logger import logger
from .parallel_input_processor import (
    ParallelInputProcessor,
    deserialize_sampling_params,
    serialize_sampling_params,
    set_env_vars,
)

_WORKER_READY = "__WORKER_READY__"


# TODO: come up with a better name.
class SelfManagedInputProcessor(ParallelInputProcessor):
    """Manages a pool of worker processes for parallel input processing.

    Uses multiprocessing.Process with Queues for communication, avoiding
    the need for module-level global state.

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
        # Use "spawn" to shield ourselves from deadlocks that could occur if threads / locks
        # are created in the main process prior to creating the worker processes.
        ctx = mp.get_context("spawn")
        self._input_queue: mp.Queue = ctx.Queue()
        self._output_queue: mp.Queue = ctx.Queue()
        self._workers: List[mp.Process] = []
        self._pending_futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._shutdown = False

        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_main,
                args=(
                    self._input_queue,
                    self._output_queue,
                    model_path,
                    tokenizer_dir,
                    checkpoint_format,
                    trust_remote_code,
                    tokenizer_mode,
                ),
                name=f"InputWorker-{i}",
            )
            p.start()
            self._workers.append(p)

        # Wait for all workers to signal readiness.
        ready_count = 0
        while ready_count < num_workers:
            msg = self._output_queue.get()
            if msg[0] == _WORKER_READY:
                ready_count += 1
                logger.debug(f"Worker {ready_count}/{num_workers} ready")

        # Start background thread that collects results and resolves futures.
        self._result_thread = threading.Thread(
            target=self._result_collector,
            daemon=True,
            name="InputProcessorResultCollector",
        )
        self._result_thread.start()

        logger.info(f"Starting parallel input processor with {num_workers} workers")

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

        task_id = str(uuid.uuid4())
        future: Future = Future()

        sampling_params_dict = serialize_sampling_params(sampling_params)

        with self._lock:
            self._pending_futures[task_id] = future

        self._input_queue.put((task_id, inputs, sampling_params_dict, use_hash))
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

        # Send poison pills to all workers.
        for _ in self._workers:
            self._input_queue.put(None)

        if wait:
            for w in self._workers:
                w.join()

        self._output_queue.put(None)
        if wait:
            self._result_thread.join()

    def _result_collector(self) -> None:
        """Background thread that receives results and resolves futures."""
        while True:
            result = self._output_queue.get()
            if result is None:
                break

            task_id, value, error = result
            with self._lock:
                future = self._pending_futures.pop(task_id, None)

            if future is not None:
                if error is not None:
                    future.set_exception(error)
                else:
                    future.set_result(value)


def _worker_main(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    model_path: str,
    tokenizer_dir: Optional[str],
    checkpoint_format: Optional[str],
    trust_remote_code: bool,
    tokenizer_mode: str,
) -> None:
    """Main entry point for worker process.

    All state is local to this function - no module-level globals needed.
    The worker runs a loop pulling tasks from input_queue, processing them,
    and putting results on output_queue.
    """
    from ..inputs import create_input_processor, create_input_processor_with_hash
    from .llm_utils import ModelLoader

    set_env_vars()

    # Initialize heavy resources (local to this worker process).
    tokenizer_path = tokenizer_dir or model_path
    tokenizer = ModelLoader.load_hf_tokenizer(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
        use_fast=(tokenizer_mode != "slow"),
    )
    input_processor = create_input_processor(
        model_path,
        tokenizer,
        checkpoint_format,
    )
    input_processor_with_hash = create_input_processor_with_hash(input_processor)

    # Signal that this worker is ready.
    output_queue.put((_WORKER_READY, None, None))

    # Main loop - pull tasks, process, return results.
    while True:
        task = input_queue.get()
        # Poison pill signals shutdown.
        if task is None:
            break

        task_id, inputs, sampling_params_dict, use_hash = task
        try:
            sampling_params = deserialize_sampling_params(sampling_params_dict)

            if use_hash and "multi_modal_data" in inputs:
                processor = input_processor_with_hash
            else:
                processor = input_processor

            prompt_token_ids, extra_processed_inputs = processor(inputs, sampling_params)
            output_queue.put((task_id, (prompt_token_ids, extra_processed_inputs), None))
        except Exception as e:
            output_queue.put((task_id, None, e))
