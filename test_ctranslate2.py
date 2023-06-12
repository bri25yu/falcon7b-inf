import ctranslate2

from test_exact import *


def run_ctranslate2() -> BenchmarkOutput:
    generator_name = model_name.split("/")[-1]

    init_time = time()
    generator = ctranslate2.Generator(generator_name, device="cuda")
    init_time = time() - init_time

    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))

    inference_time = time()
    results = generator.generate_batch([tokens], beam_size=1, max_length=max_new_tokens)
    inference_time = time() - inference_time
    output_text = tokenizer.decode(results[0].sequences_ids[0])

    return BenchmarkOutput(
        init_time=init_time,
        inference_time=inference_time,
        output_text=output_text,
    )


def run_hf(seed: int=42) -> BenchmarkOutput:
    return run_inference_on_model(init_reimpl_model, seed=seed)


def compare_ct2_hf_outputs(
    ct2_output: BenchmarkOutput, hf_output: BenchmarkOutput
) -> None:
    print(
        "Init time",
        f"\tCTranslate2 model and generation {ct2_output.init_time:.3f}s",
        f"\tHF model and generation {hf_output.init_time:.3f}s",
        sep="\n",
    )
    print(
        "Inference time",
        f"\tCTranslate2 model and generation {ct2_output.inference_time:.3f}s",
        f"\tHF model and generation {hf_output.inference_time:.3f}s",
        sep="\n",
    )

    if ct2_output.output_text != hf_output.output_text:
        print("Ctranslate2 and HF outputs DO NOT match!")
        print("Ctranslate2 output")
        print(ct2_output.output_text.removeprefix(input_text + "\n"), "\n")
        print("HF output")
        print(hf_output.output_text.removeprefix(input_text + "\n"), "\n")
    else:
        print("Ctranslate2 and HF outputs match")


if __name__ == "__main__":
    ct2_output = run_ctranslate2()
    hf_output = run_hf()
    compare_ct2_hf_outputs(
        ct2_output=ct2_output,
        hf_output=hf_output,
    )
