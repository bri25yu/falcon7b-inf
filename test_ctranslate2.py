import ctranslate2

from test_exact import *


sample_topk = 10


def run_ctranslate2() -> BenchmarkOutput:
    generator_name = model_name.split("/")[-1]

    init_time = time()
    generator = ctranslate2.Generator(generator_name, device="cuda")
    init_time = time() - init_time

    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))

    inference_time = time()
    results = generator.generate_batch([tokens], sampling_topk=sample_topk, max_length=max_new_tokens, include_prompt_in_result=True)
    inference_time = time() - inference_time
    output_text = tokenizer.decode(results[0].sequences_ids[0])

    return BenchmarkOutput(
        init_time=init_time,
        inference_time=inference_time,
        output_text=output_text,
    )


def run_hf(seed: int=42) -> BenchmarkOutput:
    init_time = time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
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
        top_k=sample_topk,
        num_return_sequences=1,
        do_sample=True,
    )[0]["generated_text"]
    inference_time = time() - inference_time

    return BenchmarkOutput(
        init_time=init_time,
        inference_time=inference_time,
        output_text=output_text,
    )


def compare_ct2_hf_outputs(
    ct2_output: BenchmarkOutput, hf_output: BenchmarkOutput
) -> None:
    print(
        "Init time",
        f"\tCTranslate2 model {ct2_output.init_time:.3f}s",
        f"\tHF best match model {hf_output.init_time:.3f}s",
        sep="\n",
    )
    print(
        "Inference time",
        f"\tCTranslate2 model {ct2_output.inference_time:.3f}s",
        f"\tHF best match model {hf_output.inference_time:.3f}s",
        sep="\n",
    )

    if ct2_output.output_text != hf_output.output_text:
        print("Ctranslate2 model output")
        print(ct2_output.output_text.removeprefix(input_text + "\n"), "\n")
        print("HF model output")
        print(hf_output.output_text.removeprefix(input_text + "\n"), "\n")
    else:
        print("Ctranslate2 and HF model outputs match")


if __name__ == "__main__":
    ct2_output = run_ctranslate2()
    hf_output = run_hf()
    compare_ct2_hf_outputs(
        ct2_output=ct2_output,
        hf_output=hf_output,
    )
