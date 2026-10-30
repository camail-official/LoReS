#!/usr/bin/env bash
usage='weights.sh: MixerZoo WikiText-2 W/A ladder and all R1/R4/Ro ablations.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
(($# == 0)) || { printf 'Unknown arguments: %s\n' "$*" >&2; exit 2; }

for cell in "${zoo[@]}"; do
    for precision in 16:16 8:8 4:16 4:8 16:4 4:4; do
        w=${precision%:*}; a=${precision#*:}
        activation=(--a_bits "$a")
        [[ $a != 4 ]] || activation+=(--a_asym)
        stacks=(000 111)
        [[ $precision != 4:4 ]] || stacks=(000 001 010 011 100 101 110 111)
        for stack in "${stacks[@]}"; do
            rotation=(--no-rotate_residual --no-rotate_r4 --no-rotate_o)
            [[ ${stack:0:1} == 0 ]] || rotation[0]=--rotate_residual
            [[ ${stack:1:1} == 0 ]] || rotation[1]=--rotate_r4
            [[ ${stack:2:1} == 0 ]] || rotation[2]=--rotate_o
            job "$cell/w${w}a${a}/$stack" "$python" -m lores ptq \
                --input_model "$models_root/mixerzoo_1.3B/$cell" --dtype bfloat16 \
                --model_max_length 2048 --eval_tasks wikitext --eval_batch_size 16 \
                --device cuda --seed 0 --rotation_seed 0 "${rotation[@]}" \
                --w_bits "$w" --w_groupsize -1 "${activation[@]}" --a_groupsize -1 \
                --nsamples 128 \
                --output-root "$output_root"
        done
    done
done
finish
