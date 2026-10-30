<h1 align="center">Quantizing recurrent language models is easier than you think</h1>

<p align="center">
  <a href="TODO-arxiv-link"><img src="https://img.shields.io/badge/Paper-B31B1B?logo=arxiv&logoColor=white" alt="Paper"></a>
</p>

**LoReS** (**Lo**w-bit **Re**current **S**tate) quantizes the recurrent state of linear-recurrent
language models and hybrid models during decoding, with value-axis scale grouping, decay-based mixed precision and
stochastic rounding.

## Installation

LoReS has two backends, each with its own environment: Hugging Face for the fake-quantized experiments and vLLM for
serving with packed INT8 state kernels. Both environments are installed with [uv](https://docs.astral.sh/uv/). Use a
recent version (tested with 0.12).

Clone the repository with its submodules and apply our patches to vLLM and flash-linear-attention:

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/camail-official/LoReS.git
cd LoReS
git -C libs/vllm apply ../../patches/vllm.patch
git -C libs/flash-linear-attention apply ../../patches/fla.patch
```

The Hugging Face backend installs into `.venv`. Building its CUDA extensions needs `nvcc` from CUDA 12. flash-attn
compiles from source, which can take more than an hour. Set `MAX_JOBS=8` if the build runs out of memory:

```bash
uv sync --frozen
```

The vLLM backend installs into `.venv-vllm` and needs `nvcc` from CUDA 13:

```bash
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --frozen --no-default-groups --group vllm
```

Tests and linters are in the optional `dev` group. Add `--group dev` to either command, for example:

```bash
uv sync --frozen --group dev
```

## Reproducing the paper

Below is one example. [`experiments/README.md`](experiments/README.md) describes all experiments.

```bash
bash experiments/evaluation/delta_ell.sh --model qwen3_5 --stage reference --run
bash experiments/evaluation/delta_ell.sh --model qwen3_5 --baseline-file <reference>/traces.pt \
    --only 'qwen3_5/INT8-value-SR-none' --run
```

## Hybrid models

[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (Gated DeltaNet),
[Nemotron-H-8B](https://huggingface.co/nvidia/Nemotron-H-8B-Reasoning-128K) (Mamba-2) and
[Kimi-Linear-48B-A3B](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) (KDA).

## Mixer zoo

1.3B models in which the sequence mixer varies while the surrounding layer design and training recipe are fixed. They
are defined in [`mixerzoo/`](mixerzoo/).

## Citation

```bibtex
TODO
```

## Dependencies

LoReS builds on [vLLM](https://github.com/vllm-project/vllm),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[Mamba](https://github.com/state-spaces/mamba), [SpinQuant](https://github.com/facebookresearch/SpinQuant),
[QuaRot](https://github.com/spcl/QuaRot), [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp),
[llm-compressor](https://github.com/vllm-project/llm-compressor),
[fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform),
[NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills) and
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Acknowledgements
This work was supported by the Hector Foundation, Max Planck ETH Center for Learning Systems, and the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

