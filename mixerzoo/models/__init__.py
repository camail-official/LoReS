from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mixer_zoo import MixerZooConfig
from .modeling_mixer_zoo import MixerZooForCausalLM, MixerZooModel
from .unified_mixer import UnifiedMixer

AutoConfig.register(MixerZooConfig.model_type, MixerZooConfig, exist_ok=True)
AutoModel.register(MixerZooConfig, MixerZooModel, exist_ok=True)
AutoModelForCausalLM.register(MixerZooConfig, MixerZooForCausalLM, exist_ok=True)
__all__ = ["MixerZooConfig", "MixerZooModel", "MixerZooForCausalLM", "UnifiedMixer"]
