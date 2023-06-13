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


def repl(
    max_new_tokens_per_step: int=100,
    stop_word: str="STOP",
    user_input_token: str="User",
    falcon_input_token: str="Falcon",
) -> None:
    print("Initializing generation...", end=" ")
    init_time = time()
    tokenizer, generator = initialize_generation()
    init_time = time() - init_time
    print(f"took {init_time:.1f}s")

    print(f"Starting chatbot. If you want to quit, please input \"{stop_word}\".\n\n")

    # TODO This is heavily unoptimized. Prompt is from https://huggingface.co/spaces/HuggingFaceH4/falcon-chat
    history = f"The following is a conversation between a highly knowledgeable and intelligent AI assistant, called {falcon_input_token}, and a human user, called {user_input_token}. In the following interactions, {user_input_token} and {falcon_input_token} will converse in natural language, and {falcon_input_token} will answer {user_input_token}'s questions. {falcon_input_token} was built to be respectful, polite and inclusive. {falcon_input_token} was built by the Technology Innovation Institute in Abu Dhabi. {falcon_input_token} will never decline to answer a question, and always attempts to give an answer that {user_input_token} would be satisfied with. It knows a lot, and always tells the truth. The conversation begins."
    try:
        while True:
            user_input = input()
            if user_input == stop_word:
                print("Goodbye!")
                break

            history += f"{user_input_token} {user_input} {falcon_input_token} "

            inference_step_time = time()
            output_text, num_output_tokens = perform_generation(tokenizer, generator, history, max_new_tokens_per_step)
            inference_step_time = time() - inference_step_time

            output_text = output_text.split(user_input_token)[0]
            output_text = output_text.strip().removesuffix(user_input_token)
            print(f"\t{output_text}\n\t({inference_step_time:.1f}s, {num_output_tokens} tokens)")

            history += output_text
    except KeyboardInterrupt:
        print("Goodbye!")
        return


if __name__ == "__main__":
    repl()
