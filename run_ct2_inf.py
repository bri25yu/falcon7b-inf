"""
First run:
```bash
python ctranslate2_falcon.py --model tiiuae/falcon-7b-instruct --quantization float16 --output_dir falcon-7b-instruct
```
"""
from typing import Tuple

from time import time

from ctranslate2 import Generator

from transformers import AutoTokenizer, PreTrainedTokenizer


def initialize_generation() -> Tuple[PreTrainedTokenizer, Generator]:
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    generator_name = model_name.split("/")[-1]
    generator = Generator(generator_name, device="cuda")

    return tokenizer, generator


def perform_generation(
    tokenizer: PreTrainedTokenizer, generator: Generator, input_text: str, max_new_tokens_per_step: int
) -> Tuple[str, int]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))
    results = generator.generate_batch([tokens], beam_size=1, max_length=max_new_tokens_per_step, include_prompt_in_result=False)
    output_ids = results[0].sequences_ids[0]
    output_text = tokenizer.decode(output_ids)

    total_num_ids = len(tokens) + len(output_ids)

    return output_text, total_num_ids


def single_turn_chat(max_new_tokens: int=300) -> None:
    print("Initializing generation...", end=" ")
    init_time = time()
    tokenizer, generator = initialize_generation()
    init_time = time() - init_time
    print(f"took {init_time:.1f}s")

    try:
        while True:
            input_text = input()
            inference_step_time = time()
            output_text, num_output_tokens = perform_generation(tokenizer, generator, input_text, max_new_tokens)
            inference_step_time = time() - inference_step_time

            print(f"\t{output_text}\n\t({inference_step_time:.1f}s, {num_output_tokens} tokens)")
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    single_turn_chat()
