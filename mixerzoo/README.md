# MixerZoo

Hugging Face model definitions for the paper's 1.3B controlled mixer comparisons.
Cells share a backbone and vary in transition, decay, write gating, and Q/K normalization.

## Layout

- `models/unified_mixer.py`: shared projections, convolutions, featurization and mixer transitions.
- `models/configuration_mixer_zoo.py`: architecture configuration.
- `models/modeling_mixer_zoo.py`: backbone and causal language model.
- `configs/`: eleven standalone cell configurations.
- `convert.py`: training DCP-to-HF checkpoint conversion.

`import mixerzoo` registers the classes with Hugging Face:

```python
import mixerzoo
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("mixerzoo/configs/kda.json")
model = AutoModelForCausalLM.from_config(config)
```

## Checkpoint conversion

```bash
uv run python -m mixerzoo.convert kda \
    --checkpoint-root /path/to/checkpoints \
    --tokenizer /path/to/tokenizer \
    --out-root /path/to/hf-models \
    --tmp-root /path/to/scratch
```

The converter reads the latest `step-*` directory beneath
`<checkpoint-root>/mixerzoo_<cell>_1.3B/checkpoint/`.
Omitting cell names converts all eleven configurations.
