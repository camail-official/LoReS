#!/usr/bin/env bash
usage='exports.sh --suite hf|vllm|speed
Refuses to overwrite an existing export.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
suite=
while (($#)); do
    case "$1" in
        --suite) suite=$2; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ $suite == hf || $suite == vllm || $suite == speed ]] || { printf '%s\n' "$usage" >&2; exit 2; }
for arch in qwen3_5 nemotronh kimi_linear; do
    prefix=$(model_field "$arch" prefix)
    if [[ $suite == hf ]]; then
        [[ $arch != kimi_linear ]] || continue
        schemes=(W4A16 W8A16)
    elif [[ $suite == speed ]]; then
        [[ $arch != kimi_linear ]] || continue
        schemes=(W4A16)
    else
        schemes=(W8A16 W8A8 W4A8)
    fi
    for scheme in "${schemes[@]}"; do
        id="$arch/${scheme,,}"
        [[ $id == $only ]] || continue
        if [[ $suite == hf ]]; then
            bits=${scheme:1:1}
            out="$models_root/${arch}_w${bits}g128_r1_llmc.pt"
            command=("$python" -m scripts.export.wa_fake_quant --arch "$arch" --w-bits "$bits" --out "$out")
        else
            out="$models_root/hf/$prefix-rot-${scheme,,}"
            command=("$python" -m scripts.export.wa_vllm --arch "$arch" --scheme "$scheme" --out "$out")
        fi
        if "$execute" && [[ -e $out ]]; then printf 'Export already exists: %s\n' "$out" >&2; exit 2; fi
        job "$id" "${command[@]}"
    done
done
finish
