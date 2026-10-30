#!/usr/bin/env bash
usage='scale_feedback.sh --suite inflation|snapshots|rounding|operands [--operands FILE]
Offline replays of the recurrent state with quantizer and oracle scales (one WikiText-2 row, 2048 tokens).
inflation: four MixerZoo models, 24 layers, both axes. snapshots: value axis, 8 layers, states at four steps;
Qwen3.5 replays the operands recorded by --suite operands. rounding: Mamba-2, eight SR draw sequences.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
suite= operands=
while (($#)); do
    case "$1" in
        --suite) suite=$2; shift 2 ;;
        --operands) operands=$2; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
cells=(mamba2_beta gated_deltanet gla_beta kda)
replay=("$python" -m scripts.diagnostics.rotation_scale_feedback --models-root "$models_root/mixerzoo_1.3B" --output-root "$output_root")
snapshots=(--layers 8 --snapshot-step 256 768 1280 1792)
case "$suite" in
    inflation)
        for cell in "${cells[@]}"; do
            for axis in value key; do job "$cell/$axis" "${replay[@]}" --cell "$cell" --axis "$axis"; done
        done ;;
    operands)
        job qwen3_5/operands "$python" -m scripts.diagnostics.record_qwen_operands \
            --revision "$(checkpoint_pin qwen3_5 revision)" --output-root "$output_root" ;;
    snapshots)
        for cell in "${cells[@]}"; do job "$cell/snapshots" "${replay[@]}" --cell "$cell" "${snapshots[@]}"; done
        if [[ -z $operands ]]; then
            if "$execute"; then printf 'Run --suite operands first, then pass its file with --operands.\n' >&2; exit 2; fi
            operands="<qwen3_5-operands>/qwen3_5_operands.pt"
        fi
        job qwen3_5/snapshots "${replay[@]}" --cell qwen3_5 --operands "$operands" "${snapshots[@]}" ;;
    rounding)
        for s in 0 1 2 3 4 5 6 7; do job "mamba2_beta/seed$s" "${replay[@]}" --cell mamba2_beta --seed $((s * 100003)); done ;;
    *) printf '%s\n' "$usage" >&2; exit 2 ;;
esac
finish
