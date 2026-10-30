#!/usr/bin/env bash
usage='residuals.sh: residual statistics/skylines and dominant-channel interventions.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
(($# == 0)) || { printf 'Unknown arguments: %s\n' "$*" >&2; exit 2; }

for cell in "${zoo[@]}"; do
    dump=()
    [[ $cell != softmax && $cell != kda ]] || dump=(--dump-activations)
    job "$cell/statistics" "$python" -m scripts.diagnostics.residuals \
        --cell "$cell" --models-root "$models_root/mixerzoo_1.3B" \
        --device cuda --seed 0 "${dump[@]}" --output-root "$output_root"
    job "$cell/interventions" "$python" -m scripts.diagnostics.channel_interventions \
        --cell "$cell" --models-root "$models_root/mixerzoo_1.3B" \
        --device cuda --seed 0 --output-root "$output_root"
done
finish
