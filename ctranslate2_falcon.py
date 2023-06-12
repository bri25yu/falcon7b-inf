import argparse

from torch import float16

from ctranslate2.specs import common_spec, transformer_spec
from ctranslate2.converters.transformers import Converter, RWLoader, TransformersConverter

from modeling_falcon import RWForCausalLM, FalconConfig


class FalconLoader(RWLoader):
    @property
    def architecture_name(self):
        return None  # This is never used in the downstream logic

    def set_linear(self, spec, module):
        spec.weight = module.weight.numpy()

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


class FalconConverter(TransformersConverter):
    # This is an exact copy of `TransformersConverter._load` unless specified otherwise
    def _load(self):
        import torch
        import transformers

        with torch.no_grad():
            ####################
            # START Skip config to get loader. use falcon loader
            ####################

            # Original code:
            # config = transformers.AutoConfig.from_pretrained(
            #     self._model_name_or_path, trust_remote_code=self._trust_remote_code
            # )

            # config_name = config.__class__.__name__
            # loader = _MODEL_LOADERS.get(config_name)

            # if loader is None:
            #     raise ValueError(
            #         "No conversion is registered for the model configuration %s "
            #         "(supported configurations are: %s)"
            #         % (config_name, ", ".join(sorted(_MODEL_LOADERS.keys())))
            #     )

            loader = FalconLoader()

            ####################
            # END Skip config to get loader. use falcon loader
            ####################

            tokenizer_class = transformers.AutoTokenizer

            kwargs = {}
            if self._load_as_float16:
                kwargs["torch_dtype"] = torch.float16
            if self._revision:
                kwargs["revision"] = self._revision
            if self._low_cpu_mem_usage:
                kwargs["low_cpu_mem_usage"] = self._low_cpu_mem_usage
            if self._trust_remote_code:
                kwargs["trust_remote_code"] = self._trust_remote_code

            ####################
            # START ignore model_class
            ####################

            # Original code:
            # model_class = getattr(transformers, loader.architecture_name)
            # model = self.load_model(model_class, self._model_name_or_path, **kwargs)

            model = self.load_model(self._model_name_or_path, **kwargs)

            ####################
            # END ignore model_class
            ####################

            tokenizer = self.load_tokenizer(
                tokenizer_class,
                self._model_name_or_path,
            )

            spec = loader(model, tokenizer)

            if self._activation_scales:
                activation_scales = torch.load(
                    self._activation_scales, map_location="cpu"
                )
                loader.smooth_activation(spec, activation_scales)

            if self._copy_files:
                for filename in self._copy_files:
                    spec.register_file(self.get_model_file(filename))

            return spec

    def load_model(self, model_name_or_path, **kwargs):
        config = FalconConfig.from_pretrained(model_name_or_path, **kwargs)
        assert config.torch_dtype == float16
        return RWForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            **kwargs,
        )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Name of the pretrained model to download, "
            "or path to a directory containing the pretrained model."
        ),
    )
    parser.add_argument(
        "--activation_scales",
        help=(
            "Path to the pre-computed activation scales. Models may "
            "use them to rescale some weights to smooth the intermediate activations "
            "and improve the quantization accuracy. See "
            "https://github.com/mit-han-lab/smoothquant."
        ),
    )
    parser.add_argument(
        "--copy_files",
        nargs="+",
        help=(
            "List of filenames to copy from the Hugging Face model to the converted "
            "model directory."
        ),
    )
    parser.add_argument(
        "--revision",
        help="Revision of the model to download from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Enable the flag low_cpu_mem_usage when loading the model with from_pretrained.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow converting models using custom code.",
    )

    Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = FalconConverter(
        args.model,
        activation_scales=args.activation_scales,
        copy_files=args.copy_files,
        load_as_float16=args.quantization in ("float16", "int8_float16"),
        revision=args.revision,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
    )
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()
