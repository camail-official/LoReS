#!/usr/bin/env bash
usage='generation_vllm.sh --suite main|cube --prompts DIRECTORY [--grader-python PATH]
main: AIME/GPQA avg@32, MMLU-Pro seed 0 in 16 complete shards.
cube: AIME FP32 and RTN/SR x symmetric/asymmetric, including Kimi key grouping.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
suite= prompts= grader="$repo/.venv/bin/python"
while (($#)); do
    case "$1" in
        --suite) suite=$2; shift 2 ;;
        --prompts) prompts=$2; shift 2 ;;
        --grader-python) grader=$2; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n $prompts && ( $suite == main || $suite == cube ) ]] || { printf '%s\n' "$usage" >&2; exit 2; }
seeds=({0..31})
for model in qwen nemo kimi; do
    tasks=(aime gpqa mmlupro) axes=(value)
    arms=(fp32 sym w8a16_fp32 w8a16_sym w8a8_fp32 w8a8_sym w4a8_fp32 w4a8_sym)  # symmetric INT8 grid, all models
    if [[ $suite == cube ]]; then
        tasks=(aime) arms=(fp32 rtn_sym rtn_asym sym asym)
        [[ $model != kimi ]] || axes=(value key)
    fi
    for task in "${tasks[@]}"; do
        for axis in "${axes[@]}"; do
            for arm in "${arms[@]}"; do
                base=("$python" -m scripts.evaluation.vllm_generate --model "$model" --task "$task" --arm "$arm" \
                    --state-axis "$axis" --models-root "$models_root/hf" --grader-python "$grader" --output-root "$output_root")
                if [[ $task == mmlupro ]]; then
                    for shard in {0..15}; do
                        job "$model/$task/$axis/$arm/$shard" "${base[@]}" --shard "$shard" --seeds 0 --prompts "$prompts/mmlupro_k$shard.jsonl"
                    done
                else
                    job "$model/$task/$axis/$arm" "${base[@]}" --seeds "${seeds[@]}" --prompts "$prompts/$task.jsonl"
                fi
            done
        done
    done
done
finish
