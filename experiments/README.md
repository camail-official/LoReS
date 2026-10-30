# Experiments

Recipes that regenerate the paper's tables and figures from pretrained checkpoints. Each recipe prints its
commands, and `--run` executes them one after another on the local machine.

Common flags: `--python PATH`, `--models-root DIR` (default `models/`), `--output-root DIR` (default `output/`),
`--only GLOB` (configuration IDs, printed above each command) and `--run`. Every run writes a new attempt directory.

`.venv` runs everything except serving. `.venv-vllm` runs the vLLM server, the kernel and serving benchmarks and
Kimi-Linear's lm-eval generation. Pass it with `--python` (with `--vllm-python` for `generation.sh`).

## Recipes

| Recipe | Paper items |
| --- | --- |
| `preparation/calibration.sh` | Decay scores, INT5 and promotion masks |
| `preparation/exports.sh` | W/A weight exports for HF and vLLM (`--suite speed`: the exports of the timing grids) |
| `preparation/prompts.sh` | AIME, GPQA and MMLU-Pro prompts |
| `evaluation/weights.sh` | MixerZoo W/A ladders and rotation ablations |
| `evaluation/delta_ell.sh` | Δℓ tables, chunk, rotation and promotion figures, rotation heatmaps (also with oracle scales) |
| `evaluation/generation.sh` | GSM8K and RULER (`--suite main`), grouping ranking, promotion and rotation transfer |
| `evaluation/generation_vllm.sh` | AIME, GPQA and MMLU-Pro (`--suite main`), quantizer cube (`--suite cube`) |
| `diagnostics/residuals.sh` | Residual statistics, outlier tables, skyline and kurtosis figures |
| `diagnostics/state_diagnostics.sh` | State geometry, state snapshots (grouping panel, one-step rotation error), error injection |
| `diagnostics/scale_feedback.sh` | State replays: scale inflation, query alignment, remaining rounding error |
| `benchmarks/speed.sh` | Kernel sweeps, batch-size and context-length decode timings |
| `analysis/*.sh` | Grading, Δℓ collection, diagnostic reductions, figures |

## Workflow

Producers print the directory of every run. Pass these paths on explicitly.

Δℓ. Hybrid models need an FP32 reference from the same code and environment. MixerZoo runs write their own. The
INT5 and promotion rows take the allocation run of `calibration.sh` with the same bit widths via `--masks`.

```bash
bash experiments/evaluation/delta_ell.sh --model qwen3_5 --stage reference --run
bash experiments/evaluation/delta_ell.sh --model qwen3_5 --baseline-file <reference>/traces.pt --run
bash experiments/evaluation/delta_ell.sh --model gated_deltanet --masks <allocation run> --run
bash experiments/analysis/collect_delta_ell.sh --selection runs.tsv --run   # lines: ID<TAB>run directory
```

Oracle-scale cells carry `-oracle` in their IDs. Collect them into their own inventory for the oracle heatmaps.

Generation and grading. Grading requires every question and seed of a configuration.

```bash
bash experiments/preparation/prompts.sh --directory output/new/prompts --run
bash experiments/evaluation/generation_vllm.sh --suite main --prompts output/new/prompts \
    --python .venv-vllm/bin/python --only 'qwen/aime/*' --run
bash experiments/analysis/grade.sh --task aime --model qwen --label fp32 --avg32 --files <run>/gen_fp32_s{0..31}.jsonl --run
```

State diagnostics. `collect_diagnostics.sh --help` lists the inputs of each kind.

```bash
bash experiments/diagnostics/state_diagnostics.sh --suite snapshots --run
bash experiments/analysis/collect_diagnostics.sh --kind oneshot --files <the eight *_states.pt> --run
bash experiments/diagnostics/scale_feedback.sh --suite inflation --run
bash experiments/analysis/collect_diagnostics.sh --kind inflation --files <the eight *_scale.pt> --run
```

Figures are written as PGF and need LaTeX. `plots.sh --help` lists the figures, and each plot module's `--help` lists its
inputs.

```bash
bash experiments/analysis/plots.sh --figure rotation-effect --directory figures --run -- --inventory inventory.json
```

## Protocol

- FP32 recurrent state, BF16 weights and activations unless W/A-quantized.
- Δℓ: seed 0, eight 2,048-token WikiText-2 sequences, readout before storage quantization, global median of the
  per-token loss differences.
- Generation: complete question sets, GSM8K with up to 8,192 generated tokens. AIME and GPQA on vLLM are averaged over
  seeds 0–31, and the promotion suite on Hugging Face uses one sample.
- W/A: llm-compressor GPTQ presets with symmetric 8-bit activations.

Reruns can differ slightly from the paper's numbers through hardware, library versions and sampling. Kimi-Linear's
value-axis R_k cells and its oracle-scale cells are not included.
