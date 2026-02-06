r"""Benchmark script comparing ParallelInputProcessor vs in-process input processing.

This script measures the performance of multimodal input processing using:
1. ParallelInputProcessor (multiprocessing-based, multiple implementations)
2. In-process input processing via create_input_processor_with_hash

Usage examples:
    # Test in-process mode
    python sandbox/benchmark_input_processor.py \
        --model-path /path/to/llava-model \
        --parallel-impl in-process \
        --num-inputs 20

    # Test parallel mode with futures (ProcessPoolExecutor)
    python sandbox/benchmark_input_processor.py \
        --model-path /path/to/llava-model \
        --parallel-impl futures --num-workers 4

    # Test with self-managed workers (mp.Process + Queues)
    python sandbox/benchmark_input_processor.py \
        --model-path /path/to/llava-model \
        --parallel-impl self-managed --num-workers 4

    # Test with mp.Pool
    python sandbox/benchmark_input_processor.py \
        --model-path /path/to/llava-model \
        --parallel-impl mp-pool --num-workers 4
"""

import argparse
import concurrent.futures
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    mode: str
    num_inputs: int
    total_time_ms: float
    latencies_ms: List[float]
    pickle_latencies_ms: Optional[List[float]] = field(default=None)
    pickle_bytes: Optional[List[int]] = field(default=None)

    @property
    def throughput(self) -> float:
        """Requests per second."""
        return self.num_inputs / (self.total_time_ms / 1000.0)

    @property
    def mean_latency_ms(self) -> float:
        return np.mean(self.latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        return np.percentile(self.latencies_ms, 50)

    @property
    def p95_latency_ms(self) -> float:
        return np.percentile(self.latencies_ms, 95)

    @property
    def p99_latency_ms(self) -> float:
        return np.percentile(self.latencies_ms, 99)

    @property
    def mean_pickle_latency_ms(self) -> Optional[float]:
        return np.mean(self.pickle_latencies_ms) if self.pickle_latencies_ms else None

    @property
    def total_pickle_time_ms(self) -> Optional[float]:
        return sum(self.pickle_latencies_ms) if self.pickle_latencies_ms else None

    @property
    def mean_pickle_bytes(self) -> Optional[float]:
        return np.mean(self.pickle_bytes) if self.pickle_bytes else None


def generate_synthetic_inputs(
    num_inputs: int, image_height: int, image_width: int, seed: int
) -> List[Dict[str, Any]]:
    """Generate reproducible synthetic multimodal inputs.

    Args:
        num_inputs: Number of inputs to generate.
        image_height: Height of synthetic images.
        image_width: Width of synthetic images.
        seed: Random seed for reproducibility.

    Returns:
        List of input dictionaries suitable for input processing.
    """
    rng = np.random.default_rng(seed)
    inputs = []
    imgs_array = rng.integers(0, 256, (num_inputs, image_height, image_width, 3), dtype=np.uint8)

    # For debugging purposes, print the first and last arrays' mean / std.
    # This should allow us to flag potential issues with e.g. reproducibility.
    first = imgs_array[0, ...]
    print(f"==== First input mean / std: {first.mean()} | {first.std()}")
    last = imgs_array[-1, ...]
    print(f"==== Last input mean / std: {last.mean()} | {last.std()}")

    for i in tqdm(range(num_inputs), desc="Generating inputs"):
        pil_image = Image.fromarray(np.ascontiguousarray(imgs_array[i]), mode="RGB")
        # pil_image = torch.from_numpy(np.ascontiguousarray(imgs_array[i]))
        input_dict = {
            "prompt": f"<image>\nDescribe this image in detail. [sample {i}]",
            "multi_modal_data": {"image": [pil_image]},
        }
        inputs.append(input_dict)

    return inputs


def run_inprocess_benchmark(
    model_path: str,
    inputs: List[Dict[str, Any]],
    warmup: int,
) -> Tuple[BenchmarkResult, Any]:
    """Run benchmark using in-process input processing.

    Args:
        model_path: Path to the multimodal model.
        inputs: List of input dictionaries to process.
        warmup: Number of warmup iterations.

    Returns:
        Tuple of (BenchmarkResult, processor) for potential reuse.
    """
    from tensorrt_llm.inputs import create_input_processor, create_input_processor_with_hash
    from tensorrt_llm.llmapi.llm_utils import ModelLoader
    from tensorrt_llm.sampling_params import SamplingParams

    print("Initializing in-process input processor...")
    tokenizer = ModelLoader.load_hf_tokenizer(model_path, trust_remote_code=True, use_fast=True)
    input_processor = create_input_processor(model_path, tokenizer, "HF")
    processor_with_hash = create_input_processor_with_hash(input_processor)

    sampling_params = SamplingParams(max_tokens=256)

    print(f"Running {warmup} warmup iterations...")
    num_warmup_iterations = min(warmup, len(inputs))
    for i in tqdm(range(num_warmup_iterations), desc="Warmup"):
        result = processor_with_hash(inputs[i], sampling_params)
        if i == 0:
            print("==== First warmup output:")
            _debug_print_result(result)
        elif i == num_warmup_iterations - 1:
            print("==== Last warmup output:")
            _debug_print_result(result)

    # Benchmark
    print(f"Running benchmark with {len(inputs)} inputs...")
    latencies = []
    pickle_latencies = []
    pickle_sizes = []
    start_total = time.perf_counter()
    for inp in tqdm(inputs, desc="Benchmark"):
        start = time.perf_counter()
        output = processor_with_hash(inp, sampling_params)
        end_process = time.perf_counter()

        pickle_start = time.perf_counter()
        pickled = pickle.dumps(output)
        pickle_end = time.perf_counter()

        latencies.append((end_process - start) * 1000)
        pickle_latencies.append((pickle_end - pickle_start) * 1000)
        pickle_sizes.append(len(pickled))
    end_total = time.perf_counter()

    total_time_ms = (end_total - start_total) * 1000
    result = BenchmarkResult(
        mode="in-process",
        num_inputs=len(inputs),
        total_time_ms=total_time_ms,
        latencies_ms=latencies,
        pickle_latencies_ms=pickle_latencies,
        pickle_bytes=pickle_sizes,
    )
    return result, processor_with_hash


def _make_parallel_processor(impl: str, model_path: str, num_workers: int):
    """Instantiate the chosen parallel input processor implementation.

    Args:
        impl: One of "futures", "self-managed", "mp-pool".
        model_path: Path to the multimodal model.
        num_workers: Number of worker processes.

    Returns:
        A ParallelInputProcessor instance.
    """
    if impl == "futures":
        from tensorrt_llm.llmapi.futures_input_processor import FuturesInputProcessor

        cls = FuturesInputProcessor
    elif impl == "self-managed":
        from tensorrt_llm.llmapi.self_managed_input_processor import SelfManagedInputProcessor

        cls = SelfManagedInputProcessor
    elif impl == "mp-pool":
        from tensorrt_llm.llmapi.pool_input_processor import MpPoolInputProcessor

        cls = MpPoolInputProcessor
    else:
        raise ValueError(f"Unknown parallel implementation: {impl}")

    return cls(
        num_workers=num_workers,
        model_path=model_path,
        checkpoint_format="HF",
        trust_remote_code=True,
        tokenizer_mode="auto",
    )


def run_parallel_benchmark(
    model_path: str,
    inputs: List[Dict[str, Any]],
    num_workers: int,
    warmup: int,
    parallel_impl: str,
) -> BenchmarkResult:
    """Run benchmark using a ParallelInputProcessor implementation.

    Args:
        model_path: Path to the multimodal model.
        inputs: List of input dictionaries to process.
        num_workers: Number of worker processes.
        warmup: Number of warmup iterations.
        parallel_impl: Which implementation to use ("futures", "self-managed", "mp-pool").

    Returns:
        BenchmarkResult with timing data.
    """
    from tensorrt_llm.sampling_params import SamplingParams

    print(f"Initializing {parallel_impl} processor with {num_workers} workers...")
    processor = _make_parallel_processor(parallel_impl, model_path, num_workers)

    sampling_params = SamplingParams(max_tokens=256)

    # Warmup - send at least num_workers requests to ensure all workers initialize.
    warmup_count = max(warmup, num_workers)
    print(f"Running {warmup_count} warmup iterations (ensuring all workers initialize)...")
    async_results = []
    for i in tqdm(range(warmup_count), desc="Submitting warmup"):
        idx = i % len(inputs)
        async_results.append(processor.process_async(inputs[idx], sampling_params, use_hash=True))

    concurrent.futures.wait(async_results)

    print("==== First warmup output:")
    _debug_print_result(async_results[0].result())
    print("==== Last warmup output:")
    _debug_print_result(async_results[-1].result())

    print(f"Running benchmark with {len(inputs)} inputs...")
    latencies = []
    async_results = []

    # NOTE: this assumes the cumulative time spent submitting requests (i.e. the total amount of time
    # spent calling `process_async`) is negligible compared to the time it takes to collect the
    # results.
    submit_start = time.perf_counter()
    for inp in tqdm(inputs, desc="Submitting requests"):
        async_results.append(processor.process_async(inp, sampling_params, use_hash=True))
        # HACK: latencies not measured per-request in parallel mode.
        latencies.append(0.1)
    submit_end = time.perf_counter()
    print(f"Submission phase took {(submit_end - submit_start) * 1000:.2f} ms")

    # Time the collection phase.
    collect_start = time.perf_counter()
    for future in tqdm(
        concurrent.futures.as_completed(async_results),
        desc="Benchmark",
        total=len(inputs),
    ):
        future.result()
    collect_end = time.perf_counter()
    print(f"Collection phase took {(collect_end - collect_start) * 1000:.2f} ms")

    end_total = time.perf_counter()

    total_time_ms = (end_total - submit_start) * 1000
    result = BenchmarkResult(
        mode=f"{parallel_impl} ({num_workers} workers)",
        num_inputs=len(inputs),
        total_time_ms=total_time_ms,
        latencies_ms=latencies,
    )

    processor.shutdown()
    return result


def print_result(result: BenchmarkResult) -> None:
    """Print benchmark results in a formatted way."""
    print(f"\n=== Results ({result.mode}) ===")
    print(f"Total time: {result.total_time_ms:.2f} ms")
    print(f"Throughput: {result.throughput:.2f} req/s")
    print("Latency (ms):")
    print(f"  Mean:  {result.mean_latency_ms:.2f}")
    print(f"  P50:   {result.p50_latency_ms:.2f}")
    print(f"  P95:   {result.p95_latency_ms:.2f}")
    print(f"  P99:   {result.p99_latency_ms:.2f}")
    if result.pickle_latencies_ms:
        print("Pickle overhead:")
        print(f"  Total pickle time: {result.total_pickle_time_ms:.2f} ms")
        print(f"  Mean pickle latency: {result.mean_pickle_latency_ms:.2f} ms")
        print(f"  Mean pickle size: {result.mean_pickle_bytes / 1024 / 1024:.2f} MB)")


def print_comparison(inprocess_result: BenchmarkResult, parallel_result: BenchmarkResult) -> None:
    """Print side-by-side comparison of results."""
    print("\n=== Comparison ===")
    print(f"{'Metric':<20} {'In-Process':>15} {'Parallel':>15} {'Speedup':>10}")
    print("-" * 62)

    speedup_total = inprocess_result.total_time_ms / parallel_result.total_time_ms
    print(
        f"{'Total time (ms)':<20} {inprocess_result.total_time_ms:>15.2f} "
        f"{parallel_result.total_time_ms:>15.2f} {speedup_total:>10.2f}x"
    )

    speedup_throughput = parallel_result.throughput / inprocess_result.throughput
    print(
        f"{'Throughput (req/s)':<20} {inprocess_result.throughput:>15.2f} "
        f"{parallel_result.throughput:>15.2f} {speedup_throughput:>10.2f}x"
    )

    speedup_mean = inprocess_result.mean_latency_ms / parallel_result.mean_latency_ms
    print(
        f"{'Mean latency (ms)':<20} {inprocess_result.mean_latency_ms:>15.2f} "
        f"{parallel_result.mean_latency_ms:>15.2f} {speedup_mean:>10.2f}x"
    )

    speedup_p50 = inprocess_result.p50_latency_ms / parallel_result.p50_latency_ms
    print(
        f"{'P50 latency (ms)':<20} {inprocess_result.p50_latency_ms:>15.2f} "
        f"{parallel_result.p50_latency_ms:>15.2f} {speedup_p50:>10.2f}x"
    )

    speedup_p95 = inprocess_result.p95_latency_ms / parallel_result.p95_latency_ms
    print(
        f"{'P95 latency (ms)':<20} {inprocess_result.p95_latency_ms:>15.2f} "
        f"{parallel_result.p95_latency_ms:>15.2f} {speedup_p95:>10.2f}x"
    )

    speedup_p99 = inprocess_result.p99_latency_ms / parallel_result.p99_latency_ms
    print(
        f"{'P99 latency (ms)':<20} {inprocess_result.p99_latency_ms:>15.2f} "
        f"{parallel_result.p99_latency_ms:>15.2f} {speedup_p99:>10.2f}x"
    )


def _debug_print_result(processor_output):
    token_ids, extra_inputs = processor_output
    print(f"---- {token_ids=}")
    if extra_inputs is not None:
        image_data = extra_inputs["multimodal_data"]["image"]
        pixel_values = image_data["pixel_values"]
        print(f"---- {pixel_values.mean()}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ParallelInputProcessor vs in-process input processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a multimodal model",
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=100,
        help="Number of workloads to process (default: 100)",
    )
    parser.add_argument(
        "--parallel-impl",
        type=str,
        required=True,
        choices=["in-process", "futures", "self-managed", "mp-pool"],
        help="Input processor implementation to benchmark",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallel modes (default: 4, ignored for in-process)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=1024,
        help="Height of synthetic images (default: 1024)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1024,
        help="Width of synthetic images (default: 1024)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Print configuration
    print("=== Benchmark Configuration ===")
    print(f"Model: {args.model_path}")
    if args.parallel_impl == "in-process":
        print("Mode: in-process")
    else:
        print(f"Mode: {args.parallel_impl} ({args.num_workers} workers)")
    print(f"Image size: {args.image_width}x{args.image_height}")
    print(f"Num inputs: {args.num_inputs}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Random seed: {args.seed}")

    # Generate synthetic inputs
    print("\nGenerating synthetic inputs...")
    inputs = generate_synthetic_inputs(
        args.num_inputs, args.image_height, args.image_width, args.seed
    )
    print(f"Generated {len(inputs)} inputs")

    if args.parallel_impl == "in-process":
        result, _ = run_inprocess_benchmark(args.model_path, inputs, args.warmup)
    else:
        result = run_parallel_benchmark(
            args.model_path, inputs, args.num_workers, args.warmup, args.parallel_impl
        )
    print_result(result)


if __name__ == "__main__":
    main()
