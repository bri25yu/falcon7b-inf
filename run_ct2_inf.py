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
) -> str:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))
    results = generator.generate_batch([tokens], beam_size=1, max_length=max_new_tokens_per_step, include_prompt_in_result=False)
    output_text = tokenizer.decode(results[0].sequences_ids[0])

    return output_text


def repl(
    max_new_tokens_per_step: int=50,
    stop_word: str="STOP",
    human_input_token: str="[|Human|]",
    ai_input_token: str="[|AI|]",
) -> None:
    print("Initializing generation...", end=" ")
    init_time = time()
    tokenizer, generator = initialize_generation()
    init_time = time() - init_time
    print(f"took {init_time:.1f}s")

    print(f"Starting chatbot. If you want to quit, please input \"{stop_word}\".\n\n")

    history = "The conversation between human and AI assistant."  # TODO This is heavily unoptimized
    try:
        while True:
            user_input = input()
            if user_input == stop_word:
                print("Goodbye!")
                break

            history += f"{human_input_token} {user_input}\n{ai_input_token} "

            inference_step_time = time()
            output_text = perform_generation(tokenizer, generator, history, max_new_tokens_per_step)
            inference_step_time = time() - inference_step_time

            output_text = output_text.split(human_input_token)[0]
            output_text = output_text.strip("\n")
            print(f"\t{output_text}\n\t({inference_step_time:.1f}s)")

            history += f"{output_text}\n"
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    repl()
