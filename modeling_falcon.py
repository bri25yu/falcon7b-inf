"""
This is a copy of https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/modelling_RW.py
with some inference optimizations.
"""
from typing import Optional, Tuple, Union
from torchtyping import TensorType

from torch.utils.checkpoint import checkpoint
from torch import LongTensor, Tensor, arange, bfloat16, cat, device as torch_device, dtype as torch_dtype, empty, float16, outer
from torch.nn import CrossEntropyLoss, Embedding, GELU, LayerNorm, Module, ModuleList, Parameter
from torch.nn.functional import dropout, scaled_dot_product_attention, linear

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from configuration_RW import RWConfig


logger = logging.get_logger(__name__)


HalfDkv = TensorType["Dkv/2"]
LHalfDkv = TensorType["L", "Dkv/2"]
LDkv = TensorType["L", "Dkv"]
DD = TensorType["Dout", "Din"]
NL = TensorType["N", "L"]
NLD = TensorType["N", "L", "D"]
_11LDkv = TensorType["1", "1", "L", "Dkv"]
NHLDkv = TensorType["N", "H", "L", "Dkv"]
NLHDkv = TensorType["N", "L", "H", "Dkv"]


# Skip weight init since we are only run finetuning or inference
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch_dtype) -> None:
        super().__init__()

        self.weight: DD = Parameter(empty((out_features, in_features), dtype=dtype))

    def forward(self, input: NLD) -> NLD:
        return linear(input, self.weight)


def apply_rotary(embeds, cos, sin):
    # embeds is NHLDkv, cos and sin are 11LDkv. output is NHLDkv
    halfDkv = embeds.size(3) // 2
    left_half, right_half = embeds[:, :, :, :halfDkv], embeds[:, :, :, halfDkv:]
    embeds_half_rotated = cat((-right_half, left_half), dim=3)

    return embeds * cos + embeds_half_rotated * sin


def rotate_half(embeds):
    # embeds is NHLDkv and output is NHLDkv
    halfDkv = embeds.size(3) // 2
    left_half, right_half = embeds[:, :, :, :halfDkv], embeds[:, :, :, halfDkv:]
    return cat((-right_half, left_half), dim=3)


class RotaryEmbedding(Module):
    def __init__(self, config: RWConfig, base: float=10000) -> None:
        super().__init__()

        Dkv = config.head_dim
        inv_freq: HalfDkv = 1.0 / (base ** (arange(0, Dkv, 2).float() / Dkv))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.seq_len_cached: int = 0
        self.initialize_cos_sin(config.custom_max_length, config.torch_dtype)

    def initialize_cos_sin(self, L: int, dtype: torch_dtype) -> None:
        if self.seq_len_cached < L:
            self.seq_len_cached = L

            t = arange(L).type_as(self.inv_freq)
            freqs: LHalfDkv = outer(t, self.inv_freq)
            emb: LDkv = cat((freqs, freqs), dim=1)

            if dtype in [float16, bfloat16]:
                emb = emb.float()

            self.cos_cached: _11LDkv = emb.cos()[None, None, :, :]
            self.sin_cached: _11LDkv = emb.sin()[None, None, :, :]

            self.cos_cached: _11LDkv = self.cos_cached.type(dtype)
            self.sin_cached: _11LDkv = self.sin_cached.type(dtype)

    def move_cos_sin_devices(self, device: torch_device) -> None:
        if device != self.cos_cached.device:
            self.cos_cached = self.cos_cached.to(device)
            self.sin_cached = self.sin_cached.to(device)

    def forward(self, query: NHLDkv, key: NHLDkv, past_key_value_length: Optional[int]=None) -> Tuple[NHLDkv, NHLDkv]:
        L = key.size(2) + (past_key_value_length if past_key_value_length is not None else 0)
        self.initialize_cos_sin(L, query.dtype)
        self.move_cos_sin_devices(query.device)

        cos, sin = self.cos_cached, self.sin_cached
        if past_key_value_length is not None:
            cos = cos[:, :, [L-1], :]
            sin = sin[:, :, [L-1], :]
        else:
            cos = cos[:, :, :L, :]
            sin = sin[:, :, :L, :]

        # return apply_rotary(query, cos, sin), apply_rotary(key, cos, sin)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        return query, key


class Attention(Module):
    def __init__(self, config: RWConfig):
        super().__init__()

        D = config.hidden_size
        H = self.H = config.n_head
        Dkv = self.Dkv = D // H
        Nkv = self.Nkv = 1

        if Dkv * H != D:
            raise ValueError(f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {D} and `num_heads`: {H}).")

        self.maybe_rotary = RotaryEmbedding(config)
        self.query_key_value = Linear(D, D + 2 * Nkv * Dkv, dtype=config.torch_dtype)
        self.dense: DD = Linear(D, D, dtype=config.torch_dtype)

    def forward(
        self,
        hidden_states: NLD,
        layer_past: Optional[Tuple[NHLDkv, NHLDkv]] = None,
        use_cache: bool = False,
    ) -> Tuple[NLD, Tuple[NHLDkv, NHLDkv]]:
        Dkv, Nkv, H = self.Dkv, self.Nkv, self.H
        N, L, _ = hidden_states.size()

        if layer_past is not None:
            using_past_key_values = True
            past_key: NHLDkv
            past_value: NHLDkv
            past_key, past_value = layer_past
            past_key_value_length = past_key.size(2)
        else:
            using_past_key_values = False
            past_key_value_length = None

        fused_qkv: NLD = self.query_key_value(hidden_states)
        fused_qkv: NLHDkv = fused_qkv.view(N, L, H + 2 * Nkv, Dkv)
        query: NHLDkv = fused_qkv[:, :, :-2 * Nkv, :].transpose(1, 2)
        key: NHLDkv = fused_qkv[:, :, -2 * Nkv: -Nkv, :].transpose(1, 2)
        value: NHLDkv = fused_qkv[:, :, -Nkv:, :].transpose(1, 2)
        query, key = self.maybe_rotary(query, key, past_key_value_length)

        if using_past_key_values:  # Concatenate after relative positional encoding to not duplicate encoding
            key: NHLDkv = cat((past_key, key), dim=2)
            value: NHLDkv = cat((past_value, value), dim=2)

        present: Tuple[NHLDkv, NHLDkv] = (key, value) if use_cache is True else None

        attn_output: NHLDkv = scaled_dot_product_attention(query, key, value, is_causal=not using_past_key_values)
        attn_output: NLHDkv = attn_output.permute(0, 2, 1, 3)
        attn_output: NLD = attn_output.reshape(N, L, H * Dkv)
        output_tensor: NLD = self.dense(attn_output)
        return output_tensor, present


class MLP(Module):
    def __init__(self, config: RWConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = Linear(hidden_size, 4 * hidden_size, dtype=config.torch_dtype)
        self.act = GELU()
        self.dense_4h_to_h = Linear(4 * hidden_size, hidden_size, dtype=config.torch_dtype)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


def dropout_add(x: Tensor, residual: Tensor, prob: float, training: bool) -> Tensor:
    out = dropout(x, p=prob, training=training)
    out = residual + out
    return out


class DecoderLayer(Module):
    def __init__(self, config: RWConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = Attention(config)

        if not config.parallel_attn:
            # unused if parallel attn
            self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = MLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

        self.config = config

    def forward(
        self,
        hidden_states: Tensor,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ):

        layernorm_output = self.input_layernorm(hidden_states)
        residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            layer_past=layer_past,
            use_cache=use_cache,
        )

        attention_output = attn_outputs[0]

        if not self.config.parallel_attn:
            residual = dropout_add(attention_output, residual, self.config.attention_dropout, training=self.training)
            layernorm_output = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        if self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class RWPreTrainedModel(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RWConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: Module):
        # Skip init since we only run finetuning or inference
        return

    def _set_gradient_checkpointing(self, module: Module, value: bool = False):
        if isinstance(module, RWModel):
            module.gradient_checkpointing = value


class RWModel(RWPreTrainedModel):
    def __init__(self, config: RWConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: Tensor):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        inputs_embeds: Optional[LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None

        for block, layer_past in zip(self.h, past_key_values):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache)

                    return custom_forward

                outputs = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    use_cache=use_cache,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
        )


class RWForCausalLM(RWPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config: RWConfig):
        super().__init__(config)
        self.transformer = RWModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, dtype=config.torch_dtype)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: NL,
        past_key_values: Optional[Tuple[Tuple[NHLDkv, NHLDkv]]] = None,
        attention_mask: Optional[NL] = None,  # Unused
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            input_ids: NL = input_ids[:, -1:]  # Get last token

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }

    def forward(
        self,
        input_ids: Optional[LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[NHLDkv, NHLDkv]], beam_idx: LongTensor
    ) -> Tuple[Tuple[NHLDkv, NHLDkv]]:
        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past_key_values for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in past_key_values
        )
        return reordered_past
