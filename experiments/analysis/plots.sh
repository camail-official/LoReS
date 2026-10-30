#!/usr/bin/env bash
usage='plots.sh --figure FIGURE --directory OUTPUT_DIRECTORY -- INPUT_FLAGS...
Figures: skyline, kurtosis, overview, injection, rotation-effect, inflation-time,
query-alignment, promotion, transfer, rotation, kernels, concurrency-bars, context, state-only.
INPUT_FLAGS are passed through to the plot module.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
figure= directory=
while (($#)); do
    case "$1" in
        --figure) figure=$2; shift 2 ;;
        --directory) directory=$2; shift 2 ;;
        --) shift; break ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n $directory ]] || { printf '%s\n' "$usage" >&2; exit 2; }
case "$figure" in
    skyline) module=weights.residual_skyline ;;
    kurtosis) module=weights.kurtosis_law ;;
    overview) module=state.state_quantization ;;
    injection) module=state.inject_rel_cdf ;;
    rotation-effect) module=state.rotation_effect ;;
    inflation-time) module=state.rotation_inflation_time ;;
    query-alignment) module=state.rotation_query_alignment ;;
    promotion) module=accuracy.promo_relative ;;
    transfer) module=accuracy.accuracy_delta_ell ;;
    rotation) module=accuracy.rank_slope_int6 ;;
    kernels) module=speed.speed_throughput ;;
    concurrency-bars) module=speed.speed_vllm_bars ;;
    context) module=speed.speed_vllm_dilution_stack ;;
    state-only) module=speed.speed_vllm_state_only ;;
    *) printf '%s\n' "$usage" >&2; exit 2 ;;
esac
job "$figure" "$python" -m "scripts.plots.$module" "$@" --output-dir "$directory"
finish
