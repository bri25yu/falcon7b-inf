from typing import Tuple

from time import time

from torch import bfloat16
from torch.nn import Module

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .modeling_falcon import RWForCausalLM


model_name = "tiiuae/falcon-7b-instruct"
max_new_tokens = 300

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

with open("lebron james.txt", encoding="utf-8") as f:
    input_text = f.read()

input_num_tokens = len(tokenizer(input_text, return_attention_mask=False).input_ids)
output_num_tokens = input_num_tokens + max_new_tokens
print(f"{input_num_tokens} input tokens and {output_num_tokens} output tokens")


def run_inference_on_model(model: Module) -> Tuple[str, float]:
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
    )

    time_taken = time()
    output_text = text_generation_pipeline(
        input_text,
        max_new_tokens=max_new_tokens,
        top_k=10,
        num_return_sequences=1,
    )[0]["generated_text"]
    time_taken = time() - time_taken

    return output_text, time_taken


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

reimpl_model = RWForCausalLM.from_pretrained(
    model_name,
    torch_dtype=bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

base_model_output, base_model_time_taken = run_inference_on_model(base_model)
reimpl_model_output, reimpl_model_time_taken = run_inference_on_model(reimpl_model)

print(f"Base model took {base_model_time_taken:.3f}s and reimpl model took {reimpl_model_time_taken:.3f}s")
if base_model_output != reimpl_model_output:
    print("Base and reimpl model outputs do not match!")
    print("Base model output", base_model_output, "Reimpl model output", reimpl_model_output, end="\n\n")
