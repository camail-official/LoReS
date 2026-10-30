#!/usr/bin/env bash
usage='state_diagnostics.sh --suite geometry|snapshots|injection
Geometry collects eight sequences at stride 64; snapshots dump one sequence at stride 8 for all eight models.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
suite=
while (($#)); do
    case "$1" in
        --suite) suite=$2; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$suite" in
    snapshots)
        for model in linear_none_beta mamba2_beta gla_beta gated_deltanet kda qwen3_5 nemotronh kimi; do
            phase=()
            [[ $model == qwen3_5 || $model == nemotronh || $model == kimi ]] || phase=(--first-step 0)
            job "$model/snapshots" "$python" -m scripts.calibration.state_geometry \
                --model "$model" --seqs 1 --seqlen 2048 --stride 8 "${phase[@]}" --dump-states \
                --models-root "$models_root/mixerzoo_1.3B" --output-root "$output_root"
        done ;;
    geometry)
        for model in linear_none_beta mamba2_beta gla_beta gated_deltanet kda qwen3_5 nemotronh kimi; do
            job "$model/geometry" "$python" -m scripts.calibration.state_geometry \
                --model "$model" --seqs 8 --seqlen 2048 --stride 64 \
                --models-root "$models_root/mixerzoo_1.3B" --output-root "$output_root"
        done ;;
    injection)
        for cell in mamba2_beta gated_deltanet; do
            job "$cell/injection" "$python" -m scripts.diagnostics.state_error_law --cell "$cell" \
                --seqs 1 --seqlen 2048 --models-root "$models_root/mixerzoo_1.3B" \
                --device cuda --output-root "$output_root" --bits 4 --inject-at 1024
        done ;;
    *) printf '%s\n' "$usage" >&2; exit 2 ;;
esac
finish
