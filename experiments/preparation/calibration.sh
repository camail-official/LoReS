#!/usr/bin/env bash
usage='calibration.sh --model MODEL --stage decay|int5|promotion [--calibration RUN_DIRECTORY]
Allocation prints its run directory. Pass it to delta_ell.sh and generation.sh with --masks.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
model= stage= calibration=
while (($#)); do
    case "$1" in
        --model) model=$2; shift 2 ;;
        --stage) stage=$2; shift 2 ;;
        --calibration) calibration=$2; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$model" in
    mamba2_beta) base=key; promoted=value; scope=per-layer; hybrid=false ;;
    gated_deltanet|gla_beta) base=value; promoted=value; scope=per-layer; hybrid=false ;;
    kda) base=key; promoted=value; scope=per-layer; hybrid=false ;;
    qwen3_5|nemotronh) base=value; promoted=value; scope=global; hybrid=true ;;
    kimi) base=key; promoted=value; scope=global; hybrid=true ;;
    *) printf '%s\n' "$usage" >&2; exit 2 ;;
esac
if [[ $stage == decay ]]; then
    if "$hybrid"; then
        job "$model/decay" "$python" -m scripts.calibration.decay_hybrid --model "$model" --steps 512 --seed 0 --output-root "$output_root"
    else
        job "$model/decay" "$python" -m scripts.calibration.decay_mixerzoo --cell "$model" --steps 512 --seed 1 \
            --models-root "$models_root/mixerzoo_1.3B" --output-root "$output_root"
    fi
else
    if [[ -z $calibration ]]; then
        if "$execute"; then printf 'Pass the decay run with --calibration.\n' >&2; exit 2; fi
        calibration="<fresh-$model-decay-run>"
    fi
    command=("$python" -m scripts.calibration.allocate --calibration "$calibration" --seed 0 --output-root "$output_root")
    if [[ $stage == int5 ]]; then
        job "$model/int5" "${command[@]}" --base-bits 4 --promoted-bits 6 --fraction 0.5 \
            --budget-scope "$scope" --base-axis "$base" --promoted-axis "$promoted"
    elif [[ $stage == promotion && ( $model == qwen3_5 || $model == nemotronh ) ]]; then
        for bits in 4 6; do
            for fraction in 0.25 0.50 0.75; do
                job "$model/$bits/$fraction" "${command[@]}" --base-bits "$bits" --promoted-bits "$((bits + 2))" \
                    --fraction "$fraction" --budget-scope per-layer --base-axis value --promoted-axis value
            done
        done
    else
        printf '%s\n' "$usage" >&2; exit 2
    fi
fi
finish
