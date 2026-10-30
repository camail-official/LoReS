#!/usr/bin/env bash
usage='delta_ell.sh --model MODEL [--stage reference|evaluate] [--baseline-file traces.pt] [--masks RUN_DIRECTORY]
Runs the configurations of configs/delta_ell.tsv. Hybrid evaluation requires a fresh reference from the same code
and environment. INT5 and promotion rows take the matching allocation run of calibration.sh with --masks.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
model= stage=evaluate baseline= masks=
while (($#)); do
    case "$1" in
        --model) model=$2; shift 2 ;;
        --stage) stage=$2; shift 2 ;;
        --baseline-file) baseline=$2; shift 2 ;;
        --masks) masks=$2; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
hybrid=false
revision=
case "$model" in
    qwen3_5|nemotronh|kimi) hybrid=true; revision=$(checkpoint_pin "$model" revision) ;;
    linear_none_beta|mamba2_beta|gla_beta|gated_deltanet|kda) ;;
    *) printf '%s\n' "$usage" >&2; exit 2 ;;
esac
if [[ $stage == reference ]]; then
    "$hybrid" || { printf 'MixerZoo writes its reference alongside each run.\n' >&2; exit 2; }
    job "$model/reference" "$python" -m scripts.evaluation.delta_ell_hybrid \
        --model "$model" --revision "$revision" --rot none --mode off --write-baseline \
        --n-seqs 8 --seqlen 2048 --prefill 1 --seed 0 --output-root "$output_root"
    finish
    exit
fi
[[ $stage == evaluate ]] || { printf 'Unknown stage: %s\n' "$stage" >&2; exit 2; }
if "$hybrid" && [[ -z $baseline ]]; then
    if "$execute"; then
        printf 'Run --stage reference first, then pass its traces.pt with --baseline-file.\n' >&2
        exit 2
    fi
    baseline="<fresh-$model-reference>/traces.pt"
fi
while IFS=$'\t' read -r id mode rotation interval clipping parity headbits chanbits mixhead scales; do
    [[ $id == "$model/"* && $id == $only ]] || continue
    extra=()
    [[ $clipping == 0 ]] || extra+=(--allow-state-clipping)
    [[ $scales == loop ]] || extra+=(--oracle-scale)
    for flag in headbits chanbits mixhead; do
        key=${!flag}
        [[ $key == - ]] && continue
        file=$(mask_file "$key" "$id")
        extra+=("--$flag" "$file")
    done
    if "$hybrid"; then
        job "$id" "$python" -m scripts.evaluation.delta_ell_hybrid \
            --model "$model" --revision "$revision" --rot "$rotation" --mode "$mode" \
            --n-seqs 8 --seqlen 2048 --prefill 1 --seed 0 \
            --baseline-file "$baseline" "${extra[@]}" --output-root "$output_root"
    else
        [[ $rotation == none ]] || extra+=("--$rotation")
        job "$id" "$python" -m scripts.evaluation.delta_ell_mixerzoo \
            --cell "$model" --models-root "$models_root/mixerzoo_1.3B" --mode "$mode" \
            --seqs 8 --seqlen 2048 --seed 0 --quant-every "$interval" --parity-len "$parity" \
            --tag "${id#*/}" "${extra[@]}" --output-root "$output_root"
    fi
done < "$repo/experiments/configs/delta_ell.tsv"
finish
