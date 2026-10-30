from transformers.configuration_utils import PretrainedConfig


class MixerZooConfig(PretrainedConfig):
    model_type = "mixer_zoo"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 24,
        num_heads: int = 4,
        head_dim: int = 128,
        mixer_family: str = "delta",
        decay: str = "diagonal",
        gate_low_rank_dim: int = 128,
        gate_logit_normalizer: int = 16,
        rope_theta: float = 10000.0,
        qk_norm: str = "l2",
        linear_write_gate: bool = False,
        conv_size: int = 4,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        vocab_size: int = 32000,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mixer_family = mixer_family
        self.decay = decay
        self.gate_low_rank_dim = gate_low_rank_dim
        self.gate_logit_normalizer = gate_logit_normalizer
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.linear_write_gate = linear_write_gate
        self.conv_size = conv_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.vocab_size = vocab_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
