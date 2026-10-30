#!/usr/bin/env bash
usage='grade.sh --task gsm8k|aime|gpqa|mmlupro|ruler --model MODEL --label LABEL [--avg32] --files FILE...
Pass ONLY files from one selected configuration. RULER takes 13 metric JSONs;
other tasks take response files. Full question/seed coverage is mandatory.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
task= model= label= seeds=(0) files=()
while (($#)); do
    case "$1" in
        --task) task=$2; shift 2 ;;
        --model) model=$2; shift 2 ;;
        --label) label=$2; shift 2 ;;
        --avg32) seeds=({0..31}); shift ;;
        --files) shift; files=("$@"); break ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n $task && -n $model && -n $label && ${#files[@]} -gt 0 ]] || { printf '%s\n' "$usage" >&2; exit 2; }
case "$task" in
    gsm8k|aime|gpqa|mmlupro|ruler) ;;
    *) printf 'Unknown task: %s\n' "$task" >&2; exit 2 ;;
esac
mode=()
[[ $task != aime ]] || mode=(--grade-mode ns)
job "$model/$task/$label" "$python" -m "scripts.grading.grade_$task" "${mode[@]}" --model "$model" \
    --label "$label" --seeds "${seeds[@]}" --files "${files[@]}" --output-root "$output_root"
finish
