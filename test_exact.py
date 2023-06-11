from typing import Callable

from time import time

from dataclasses import dataclass

from torch import bfloat16
from torch.nn import Module

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed

from modeling_falcon import RWForCausalLM


model_name = "tiiuae/falcon-7b-instruct"
max_new_tokens = 300

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

with open("lebron james.txt", encoding="utf-8") as f:
    input_text = f.read()

input_num_tokens = len(tokenizer(input_text, return_attention_mask=False).input_ids)
output_num_tokens = input_num_tokens + max_new_tokens
print(f"{input_num_tokens} input tokens and {output_num_tokens} output tokens")


@dataclass
class BenchmarkOutput:
    init_time: float  # secs
    inference_time: float  # secs
    output_text: str


def run_inference_on_model(model_init: Callable[[], Module], seed: int=42) -> BenchmarkOutput:
    init_time = time()
    model = model_init()
    init_time = time() - init_time

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
    )

    set_seed(seed)
    inference_time = time()
    output_text = text_generation_pipeline(
        input_text,
        max_new_tokens=max_new_tokens,
        top_k=10,
        num_return_sequences=1,
    )[0]["generated_text"]
    inference_time = time() - inference_time

    return BenchmarkOutput(
        init_time=init_time,
        inference_time=inference_time,
        output_text=output_text,
    )


def init_base_model() -> Module:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )


def init_reimpl_model() -> Module:
    return RWForCausalLM.from_pretrained(
        model_name,
        torch_dtype=bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

base_output = run_inference_on_model(init_base_model)
reimpl_output = run_inference_on_model(init_reimpl_model)

print(
    "Init time",
    f"\tBase model {base_output.init_time:.3f}s",
    f"\tReimpl model {reimpl_output.init_time:.3f}s",
    sep="\n",
)
print(
    "Inference time",
    f"\tBase model {base_output.inference_time:.3f}s",
    f"\tReimpl model {reimpl_output.inference_time:.3f}s",
    sep="\n",
)

if base_output.output_text != reimpl_output.output_text:
    print("Base and reimpl model outputs do not match!")
    print("Base model output", base_output.output_text, "Reimpl model output", reimpl_output.output_text, sep="\n\n")
