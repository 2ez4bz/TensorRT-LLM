import statistics
import time

from transformers import AutoProcessor
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs

from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    _install_processor_output_validation_filter,
)

MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"
KWARGS = {"padding": True, "do_rescale": False, "return_tensors": "pt", "video_metadata": None}
N = 300
ROUNDS = 5

processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
init_kwargs = processor.tokenizer.init_kwargs


def measure():
    for _ in range(20):
        processor._merge_kwargs(Qwen3VLProcessorKwargs, tokenizer_init_kwargs=init_kwargs, **KWARGS)
    t0 = time.perf_counter()
    for _ in range(N):
        processor._merge_kwargs(Qwen3VLProcessorKwargs, tokenizer_init_kwargs=init_kwargs, **KWARGS)
    return (time.perf_counter() - t0) / N * 1e6


before = [measure() for _ in range(ROUNDS)]
_install_processor_output_validation_filter()
after = [measure() for _ in range(ROUNDS)]

print("before_us", [round(x, 1) for x in before])
print("after_us", [round(x, 1) for x in after])
print("before_median_us", round(statistics.median(before), 1))
print("after_median_us", round(statistics.median(after), 1))
print("speedup", round(statistics.median(before) / statistics.median(after), 1))
