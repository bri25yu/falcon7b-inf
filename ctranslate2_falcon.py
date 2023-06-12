from ctranslate2.specs import common_spec, transformer_spec

from ctranslate2.converters.transformers import RWLoader, register_loader


@register_loader("RWConfig")
class FalconLoader(RWLoader):
    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.n_layer,
            model.config.n_head,
            pre_norm=True,
            activation=common_spec.Activation.GELU,
            rotary_dim=0,
            rotary_interleave=False,
            parallel_residual=True,  # Use parallen attention
            shared_layer_norm=True,
            multi_query_attention=True,  # Use multi-query
        )

        self.set_decoder(spec.decoder, model.transformer)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.word_embeddings)
        self.set_layer_norm(spec.layer_norm, module.ln_f)

        for layer_spec, layer in zip(spec.layer, module.h):
            if hasattr(layer_spec, "shared_layer_norm"):
                self.set_layer_norm(layer_spec.shared_layer_norm, layer.input_layernorm)
            else:
                raise ValueError("Falcon implementation should have a shared layernorm for parallel attention")

            self.set_linear(
                layer_spec.self_attention.linear[0],
                layer.self_attention.query_key_value,
            )
            self.set_linear(
                layer_spec.self_attention.linear[1], layer.self_attention.dense
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.dense_h_to_4h)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.dense_4h_to_h)

    def set_qkv_linear(self, spec, module, num_heads):
        weight = module.weight
        weight = weight.reshape(num_heads, 3, -1, weight.shape[-1])
        weight = weight.transpose(0, 1)
        weight = weight.reshape(-1, weight.shape[-1])
        spec.weight = weight.numpy()
