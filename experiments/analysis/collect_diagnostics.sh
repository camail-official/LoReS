#!/usr/bin/env bash
usage='collect_diagnostics.sh --kind grouping|injection|oneshot|inflation|alignment|rounding --files FILE...
grouping: Mamba-2 then GDN <model>_states.pt snapshots (state_diagnostics.sh --suite snapshots); writes
  row_error_vs_magnitude_<model>.json (fig:sq-overview panel b, plots.sh overview --grouping).
injection: their two .pt traces.
oneshot: the eight <model>_states.pt snapshots; one-step rotation error per model (tab:sq-oneshot-rot).
inflation: the eight replays of scale_feedback.sh --suite inflation (rotation figure b, inflation over time).
alignment: the five replays of scale_feedback.sh --suite snapshots (query-alignment figures).
rounding: the eight replays of scale_feedback.sh --suite rounding, seed 0 first (fig:sq-overview panel d).'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
kind= files=()
while (($#)); do
    case "$1" in
        --kind) kind=$2; shift 2 ;;
        --files) shift; files=("$@"); break ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$kind/${#files[@]}" in
    grouping/2)
        out="$output_root/new/diagnostics/row-error-vs-magnitude/run-$(date +%Y%m%d-%H%M%S)"
        job grouping/mamba2_beta "$python" -m scripts.diagnostics.row_error_vs_magnitude \
            --model mamba2_beta --snapshot "${files[0]}" --bits 8 --out "$out"
        job grouping/gated_deltanet "$python" -m scripts.diagnostics.row_error_vs_magnitude \
            --model gated_deltanet --snapshot "${files[1]}" --bits 8 --out "$out" ;;
    injection/2)
        job injection "$python" -m scripts.diagnostics.error_decay --sequence 0 --traces "${files[@]}" --output-root "$output_root" ;;
    oneshot/8)
        for file in "${files[@]}"; do
            model=$(basename -- "$file" _states.pt)
            job "oneshot/$model" "$python" -m scripts.diagnostics.oneshot_rotation_error \
                --model "$model" --snapshot "$file" --output-root "$output_root"
        done ;;
    inflation/8|alignment/5|rounding/8)
        job "$kind" "$python" -m scripts.diagnostics.reduce_scale_feedback --kind "$kind" --files "${files[@]}" \
            --output-root "$output_root" ;;
    *) printf '%s\n' "$usage" >&2; exit 2 ;;
esac
finish
