import argparse

import torch
from PIL import Image

MODEL_PATH = "/home/scratch.williamz_gpu/checkpoints/nano_v3_omni_02_03_bis"
IMAGE_PATHS = [
    "/home/scratch.williamz_gpu/code/TensorRT-LLM/sandbox/Walleposter.jpg",
    "/home/scratch.williamz_gpu/code/TensorRT-LLM/sandbox/cars.jpg",
]
PROMPT = "What are the titles of these 2 movies?"
TEXT_PROMPT = "Voila, in view, a humble vaudevillian "
MAX_TOKENS = 128


def load_images():
    return [Image.open(path).convert("RGB") for path in IMAGE_PATHS]


@torch.no_grad()
def baseline():
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.bfloat16,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    images = load_images()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"==== {prompt=}")
    inputs = processor(text=[prompt], images=images, return_tensors="pt").to(device)

    generated_ids = model.generate(
        pixel_values=inputs.pixel_values,
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"Output: {output_text[0]}")


def trtllm():
    from transformers import AutoProcessor

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    llm = LLM(
        model=MODEL_PATH,
        backend="pytorch",
        kv_cache_config=KvCacheConfig(enable_block_reuse=False),
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    images = load_images()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"==== {prompt=}")

    inputs = [{"prompt": prompt, "multi_modal_data": {"image": images}}]

    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    outputs = llm.generate(inputs, sampling_params)

    print(f"Output: {outputs[0].outputs[0].text}")


def vllm():
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL_PATH, trust_remote_code=True)

    images = load_images()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": images[0]},
                {"type": "image_pil", "image_pil": images[1]},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]

    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    outputs = llm.chat(messages=messages, sampling_params=sampling_params)

    print(f"Output: {outputs[0].outputs[0].text}")


def baseline_text_only():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.bfloat16,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    inputs = tokenizer(TEXT_PROMPT, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"Output: {output_text[0]}")


def main():
    parser = argparse.ArgumentParser(description="Nano V3 Omni tinker script")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("baseline", help="Run with HuggingFace transformers")
    subparsers.add_parser("baseline_text_only", help="Run with HuggingFace transformers")
    subparsers.add_parser("trtllm", help="Run with TensorRT-LLM")
    subparsers.add_parser("vllm", help="Run with TensorRT-LLM")

    args = parser.parse_args()

    if args.command == "baseline":
        baseline()
    if args.command == "baseline_text_only":
        baseline_text_only()
    elif args.command == "trtllm":
        trtllm()
    elif args.command == "vllm":
        vllm()


if __name__ == "__main__":
    main()
