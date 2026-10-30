#!/usr/bin/env bash
usage='speed.sh --suite tune-nemo|tune-qwen|select-nemo|select-qwen|kernels-h100|kernels-rtx|concurrency|context
W16A16/W8A16/W4A16, FP32 vs INT8-SR state, fixed-batch decode; DAO W/A rotations.
Flags: --state-fast --decode-tuning PATH --kernel-run PATH --batches N
       --concurrencies "1 2 4 ... 256" (fig:speed-vllm-joint a uses bf16 up to 256)
       --no-wa-rot (timing control without online W/A rotations; fig:speed-vllm-w8-no-rot: context, w8)'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
suite=
tuning=
kernel_run=
batches=16
state_fast=false
state_args=()
mode=nonfast
concurrencies=
no_wa_rot=false
while (($#)); do
    case "$1" in
        --suite) suite=$2; shift 2 ;;
        --decode-tuning) tuning=$2; shift 2 ;;
        --kernel-run) kernel_run=$2; shift 2 ;;
        --batches) batches=$2; shift 2 ;;
        --state-fast) state_fast=true; state_args=(--state-fast); mode=fast; shift ;;
        --concurrencies) concurrencies=$2; shift 2 ;;
        --no-wa-rot) no_wa_rot=true; mode="$mode-norot"; shift ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$suite" in
    tune-nemo|tune-qwen)
        job "$suite/$mode" "$python" -m scripts.benchmarks.vllm_kernels \
            --profile "${suite#tune-}" --sweep --gpu-tag h100 --output-root "$output_root" "${state_args[@]}" ;;
    select-nemo|select-qwen)
        if [[ -z $kernel_run ]]; then
            printf 'Full-model selection requires --kernel-run with completed isolated candidates.\n' >&2
            exit 2
        fi
        job "$suite/$mode" "$python" -m scripts.benchmarks.serving_tuning \
            --model "${suite#select-}" --kernel-run "$kernel_run" --batches "$batches" \
            --models-root "$models_root/hf" --output-root "$output_root" "${state_args[@]}" ;;
    kernels-h100|kernels-rtx)
        if "$state_fast"; then
            printf 'Equal-state kernel sweeps retain the non-FAST recipe.\n' >&2
            exit 2
        fi
        for arch in gdn kda mamba2; do
            job "$suite/$arch" "$python" -m scripts.benchmarks.vllm_kernels \
                --profile equistate --arch "$arch" --sweep --gpu-tag "${suite#kernels-}" --output-root "$output_root"
        done ;;
    concurrency|context)
        if [[ -z $tuning ]]; then
            if "$state_fast"; then
                printf 'FAST serving requires --decode-tuning with a matching FAST-on table.\n' >&2
                exit 2
            fi
            tuning="$repo/experiments/configs/serving_tuning.json"
        fi
        for model in nemo qwen; do
            for weight in bf16 w8 w4; do
                for pinned in no yes; do
                    [[ $pinned == no || ( $suite == concurrency && $weight == w4 && $state_fast == true ) ]] || continue
                    extra=()
                    [[ $pinned == no ]] || extra+=(--pinned)
                    [[ -z $concurrencies ]] || extra+=(--concurrencies $concurrencies)
                    ! "$no_wa_rot" || extra+=(--no-wa-rot)
                    job "$suite/$model/$weight/$mode/$pinned" "$python" -m scripts.benchmarks.serving \
                        --sweep "$suite" --model "$model" --weight "$weight" --repetitions 7 --batches "$batches" \
                        --gpu-tag h100 --decode-tuning "$tuning" --models-root "$models_root/hf" \
                        --output-root "$output_root" "${state_args[@]}" "${extra[@]}"
                done
            done
        done ;;
    *) printf '%s\n' "$usage" >&2; exit 2 ;;
esac
finish
