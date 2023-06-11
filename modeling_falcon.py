"""
This is a copy of https://huggingface.co/tiiuae/falcon-7b-instruct/raw/main/modelling_RW.py
with some changes to improve inference speed. 
"""
from typing import Optional, Tuple, Union
from torchtyping import TensorType

from torch import LongTensor, Tensor, arange, bfloat16, cat, empty, float16, outer
from torch.utils import checkpoint
from torch.nn import CrossEntropyLoss, Embedding, GELU, LayerNorm, Module, ModuleList, Parameter
from torch.nn.functional import dropout, scaled_dot_product_attention
from torch.jit import script

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from configuration_RW import RWConfig


logger = logging.get_logger(__name__)


NLD = TensorType["N", "L", "D"]
DD = TensorType["D_out", "D_in"]
_1LDkv = TensorType["1", "L", "Dkv"]  # Names in Python cannot start with a numeral :(
L = TensorType["L"]
HalfDkv = TensorType["Dkv/2"]
LHalfDkv = TensorType["L", "Dkv/2"]
LDkv = TensorType["L", "Dkv"]
NLHDkv = TensorType["N", "L", "H", "Dkv"]
NHLDkv = TensorType["N", "H", "L", "Dkv"]


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, dtype=None) -> None:
        super().__init__()

        # We use empty here to skip one round of parameter initialization
        # Since this module is only used for inference or finetuning a pretrained model at the moment
        self.weight: DD = Parameter(empty((out_features, in_features), dtype=dtype))

    def forward(self, input: NLD) -> NLD:
        """
        TODO
        This is from the falcon implementation. Not sure if this is
        more or less efficient that using the following:

        from torch.nn.functional import linear
        linear(input, self.weight)
        """
        return input @ self.weight.T


@script
def apply_rotary(embeds, cos, sin):
    # embeds is NLDkv, cos and sin are 1LDkv. output is NLDkv
    # Handle a possible sequence length mismatch in between q and k
    L = embeds.size(1)
    cos = cos[:, :L, :]
    sin = sin[:, :L, :]

    # left_half and right_half are NLHalfDkv. embeds_half_rotated is NLDkv
    left_half, right_half = embeds.chunk(2, dim=2)  # In the D dimension
    embeds_half_rotated = cat((-right_half, left_half), dim=2)

    return embeds * cos + embeds_half_rotated * sin


class RotaryEmbedding(Module):
    def __init__(self, head_dim: int, base: float=10000) -> None:
        super().__init__()

        inv_freq: HalfDkv = 1.0 / (base ** (arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.seq_len_cached: Optional[int] = None
        self.cos_cached: Optional[_1LDkv] = None
        self.sin_cached: Optional[_1LDkv] = None

    def cos_sin(self, seq_len: int, device="cuda", dtype=bfloat16) -> Tensor:
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t: L = arange(seq_len, device=device).type_as(self.inv_freq)

            freqs: LHalfDkv = outer(t, self.inv_freq)
            emb: LDkv = cat((freqs, freqs), dim=-1).to(device)

            if dtype in [float16, bfloat16]:
                emb = emb.float()

            self.cos_cached: _1LDkv = emb.cos()[None, :, :]
            self.sin_cached: _1LDkv = emb.sin()[None, :, :]

            self.cos_cached = self.cos_cached.type(dtype)
            self.sin_cached = self.sin_cached.type(dtype)

        return self.cos_cached, self.sin_cached

    def forward(self, query: NLD, key: NLD) -> Tuple[NLD, NLD]:
        seq_len = query.size(1)
        cos, sin = self.cos_sin(seq_len, query.device, query.dtype)
        return apply_rotary(query, cos, sin), apply_rotary(key, cos, sin)


class Attention(Module):
    def __init__(self, config: RWConfig) -> None:
        super().__init__()

        self.num_heads = config.n_head
        self.head_dim = config.hidden_size // self.num_heads

        if self.head_dim * self.num_heads != config.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {config.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.maybe_rotary = RotaryEmbedding(config.head_dim)

        self.query_key_value = Linear(
            config.hidden_size,
            config.hidden_size + 2 * self.head_dim,
            dtype=config.torch_dtype,
        )
        self.dense = Linear(config.hidden_size, config.hidden_size, dtype=config.torch_dtype)
        self.num_kv = 1

    def forward(
        self,
        hidden_states: NLD,
        layer_past: Optional[Tuple[Tensor, Tensor]]=None,
        use_cache: bool=False,
    ) -> Tuple[NLD, Optional[Tuple[NLD, NLD]]]:
        N, L, _ = hidden_states.size()
        H, Dkv, Nkv = self.num_heads, self.head_dim, self.num_kv

        fused_qkv: NLD = self.query_key_value(hidden_states)
        fused_qkv: NLHDkv = fused_qkv.view(N, L, H + 2, Dkv)

        query: NLHDkv = fused_qkv[:, :, :-2, :]
        key: NLHDkv = fused_qkv[:, :, [-2], :]
        value: NLHDkv = fused_qkv[:, :, [-1], :]

        def reshape(t: NLHDkv) -> NLD:
            return t.transpose(1, 2).reshape(-1, L, Dkv)

        query, key, value = list(map(reshape, (query, key, value)))
        query, key = self.maybe_rotary(query, key)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            # TODO Why is the key not actually concatenated along the seq len
            key = cat((past_key, key), dim=1)
            value = cat((past_value, value), dim=1)

        present: Tuple[NLD, NLD] = (key, value) if use_cache is True else None

        """
        TODO
        Sad that we have to do a reshape here. This is because
        - rotary embeddings input/output shapes. easy fix
        - kv cache. medium fix
        """
        query: NHLDkv = query.reshape(N, H, -1, Dkv)
        key: NHLDkv = key.reshape(N, Nkv, -1, Dkv)
        value: NHLDkv = value.reshape(N, Nkv, -1, Dkv)

        attn_output: NHLDkv = scaled_dot_product_attention(
            query, key, value, None, 0.0, is_causal=True
        )

        attn_output: NHLDkv = attn_output.view(N, H, L, Dkv)
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
        """
        We skip all weight inits since we'll be calling
        model.load_state_dict() anyways.
        """
        return

    def _set_gradient_checkpointing(self, module: Module, value: bool = False):
        if isinstance(module, RWModel):
            module.gradient_checkpointing = value

    @staticmethod
    def _convert_to_standard_cache(
        past_key_value: Tuple[Tuple[Tensor, Tensor]], batch_size: int
    ) -> Tuple[Tuple[Tensor, Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def _convert_to_rw_cache(
        past_key_value: Tuple[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tuple[Tensor, Tensor]]:
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )


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
        **kwargs,
    ) -> Union[Tuple[Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

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
        input_ids: LongTensor,
        past: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to our's format if needed
            if past[0][0].shape[0] == input_ids.shape[0]:
                past = self._convert_to_rw_cache(past)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
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
        **kwargs,
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
        self, past: Tuple[Tuple[Tensor, Tensor], ...], beam_idx: LongTensor
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_rw_cache(reordered_past)
