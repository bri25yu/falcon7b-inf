"""
This is a copy of https://huggingface.co/tiiuae/falcon-7b-instruct/raw/main/configuration_RW.py. 
"""
from transformers.configuration_utils import PretrainedConfig


class FalconConfig(PretrainedConfig):
    model_type = "RefinedWebModel"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        match_baseline_rotary: bool=False,
        **kwargs,
    ):
        # Unused kwargs
        kwargs_to_remove = [
            "n_embed",  # Backward compatibility with n_embed kwarg
            "apply_residual_connection_post_layernorm",  # default to false
            "initializer_range",  # We apply no weight inits
            "alibi",  # We default to rotary
            "parallel_attn",  # Default to true
            "bias",  # Default to false
            "multi_query",  # Default to true
            "hidden_dropout",  # Default to 0.0
            "attention_dropout",  # Default to 0.0, never used
        ]
        for kwarg_to_remove in kwargs_to_remove:
            if kwarg_to_remove in kwargs:
                kwargs.pop(kwarg_to_remove)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.pad_token_id = eos_token_id
        self.eos_token_id = eos_token_id

        # TODO this is in place to match the baseline rotary embedding cache initialization
        self.match_baseline_rotary = match_baseline_rotary

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=self.pad_token_id, **kwargs
        )

    @property
    def head_dim(self):
        return self.hidden_size // self.n_head

    @property
    def custom_max_length(self):
        return 1500  # hardcoded for now
