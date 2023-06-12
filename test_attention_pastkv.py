from torch import allclose, bfloat16, no_grad, rand

from transformers import set_seed

from configuration_falcon import FalconConfig
from modeling_falcon import Attention, RotaryEmbedding


model_name = "tiiuae/falcon-7b-instruct"
N, L, = 1, 100

device = "cuda"
dtype = bfloat16

set_seed(42)
total_dummy_inputs = rand(N, L, 4544, dtype=dtype, device=device)
dummy_inputs = total_dummy_inputs[:, :-1, :]
next_dummy_input = total_dummy_inputs[:, -1:, :].clone()

with no_grad():
    reimpl_config = FalconConfig.from_pretrained(model_name, torch_dtype=dtype)
    reimpl_rotary = RotaryEmbedding(reimpl_config).to(device)
    reimpl_attention = Attention(reimpl_config, reimpl_rotary).to(device)
    reimpl_attention.query_key_value.weight.normal_()
    reimpl_attention.dense.weight.normal_()

    _, past_key_values = reimpl_attention(dummy_inputs, use_cache=True)

    next_base_outputs, next_base_past_key_values = reimpl_attention(total_dummy_inputs, use_cache=True)
    next_reimpl_output, next_reimpl_past_key_values = reimpl_attention(next_dummy_input, use_cache=True, past_key_values=past_key_values)

assert allclose(next_base_past_key_values[0][:, :, :-1, :], next_reimpl_past_key_values[0][:, :, :-1, :])  # Passes?
assert allclose(next_base_past_key_values[1][:, :, :-1, :], next_reimpl_past_key_values[1][:, :, :-1, :])  # Passes?
assert allclose(next_base_past_key_values[0][:, :, -1:, :], next_reimpl_past_key_values[0][:, :, -1:, :])  # Passes?
assert allclose(next_base_past_key_values[1][:, :, -1:, :], next_reimpl_past_key_values[1][:, :, -1:, :])  # Passes?
assert allclose(next_base_outputs[:, -1:, :], next_reimpl_output)  # Fails
