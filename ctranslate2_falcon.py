@register_loader("RWConfig")
class RWLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "AutoModelForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.n_layer,
            model.config.n_head,
            pre_norm=True,
            activation=common_spec.Activation.GELU,
            alibi=model.config.alibi,
            alibi_use_positive_positions=True,
            rotary_dim=0,
            rotary_interleave=False,
            parallel_residual=model.config.parallel_attn,
            shared_layer_norm=True,
            multi_query_attention=model.config.multi_query,
        )

        self.set_decoder(spec.decoder, model.transformer)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.eos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.eos_token

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.word_embeddings)
        self.set_layer_norm(spec.layer_norm, module.ln_f)

        for layer_spec, layer in zip(spec.layer, module.h):
            if hasattr(layer_spec, "shared_layer_norm"):
                self.set_layer_norm(layer_spec.shared_layer_norm, layer.input_layernorm)
            else:
                self.set_layer_norm(
                    layer_spec.self_attention.layer_norm, layer.input_layernorm
                )
                self.set_layer_norm(
                    layer_spec.ffn.layer_norm, layer.post_attention_layernorm
                )

            if layer.self_attention.multi_query:
                self.set_linear(
                    layer_spec.self_attention.linear[0],
                    layer.self_attention.query_key_value,
                )
            else:
                self.set_qkv_linear(
                    layer_spec.self_attention.linear[0],
                    layer.self_attention.query_key_value,
                    layer.self_attention.num_heads,
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

        if module.bias is not None:
            bias = module.bias
            bias = bias.reshape(num_heads, 3, -1)
            bias = bias.transpose(0, 1)
            bias = bias.reshape(-1)
            spec.bias = bias.numpy()
