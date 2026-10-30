#!/usr/bin/env bash
usage='generation.sh --suite main|ranking|promotion|rotation [--vllm-python PATH  (default: .venv-vllm/bin/python)] [--masks RUN_DIRECTORY]
main: GSM8K and RULER-32k, Qwen/Nemotron on HF, Kimi on vLLM. Other suites use HF. All questions are evaluated.
promotion: the promoted rows take the matching allocation run of calibration.sh with --masks.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
suite= vllm_python="$repo/.venv-vllm/bin/python" masks=
while (($#)); do
    case "$1" in
        --suite) suite=$2; shift 2 ;;
        --vllm-python) vllm_python=$2; shift 2 ;;
        --masks) masks=$2; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ $suite == main || $suite == ranking || $suite == promotion || $suite == rotation ]] || { printf '%s\n' "$usage" >&2; exit 2; }
ruler=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue ruler_vt ruler_cwe ruler_fwe ruler_qa_squad ruler_qa_hotpot)

emit() {
    local label=$1 task=$2 mode=$3 axis=$4 rot=$5 w=$6 a=$7
    shift 7
    local id="$model/$task/$label" cap=65536 sampling=do_sample=True,temperature=0.6,top_p=0.95
    [[ $id == $only ]] || return 0
    local extra=("$@") tasks=$task interpreter=$python
    if [[ $model == qwen3_5 ]]; then sampling+=,top_k=20
    elif [[ $model == nemotronh ]]; then sampling+=,top_k=50
    else sampling=do_sample=True,temperature=1.0,top_p=0.95; fi
    if [[ $task == gsm8k_cot ]]; then cap=8192; fi
    if [[ $task == aime ]]; then tasks=aime24_ns,aime25_ns; fi
    if [[ $task == niah_* || $task == ruler_* ]]; then
        extra+=(--reasoning-off --task-metadata "{\"pretrained\":\"$checkpoint\",\"max_seq_lengths\":[32768]}" --max-length 34816 --attn-impl sdpa --limit 500)
    else
        sampling+=,max_gen_toks=$cap
    fi
    if [[ $model == kimi ]]; then
        interpreter=$vllm_python
        local tp=2
        if ((w < 16)); then
            tp=1
            extra+=(--vllm-ckpt "$models_root/hf/$export_prefix-rot-w${w}a$a")
        fi
        extra+=(--backend vllm --vllm-kwargs "{\"tensor_parallel_size\":$tp,\"mamba_ssm_cache_dtype\":\"float32\",\"max_model_len\":34816,\"gpu_memory_utilization\":0.9}")
    else
        extra+=(--backend hf --fp32-state)
        if ((w < 16)); then
            local weights="$models_root/${model}_w${w}g128_r1_llmc.pt"
            if "$execute" && [[ ! -f $weights ]]; then printf 'First run exports.sh --suite hf: %s\n' "$weights" >&2; exit 2; fi
            extra+=(--w-file "$weights")
        fi
    fi
    local batch=8
    [[ $task != gsm8k_cot ]] || batch=42
    [[ $task != aime ]] || batch=15
    if [[ $task == niah_* || $task == ruler_* ]]; then
        batch=1
        [[ $model != qwen3_5 ]] || batch=4
    fi
    [[ $model != kimi ]] || batch=auto
    job "$id" "$interpreter" -m scripts.evaluation.generate --model "$model" --rot "$rot" --mode "$mode" --state-axis "$axis" \
        --tasks "$tasks" --stem "${task}_${suite}_${label}" --batch-size "$batch" --w-bits "$w" --a-bits "$a" \
        --apply-chat-template --include-path "$repo/lores/lmeval_tasks" --gen-kwargs "$sampling" \
        "${extra[@]}" --output-root "$output_root"
}

for model in qwen3_5 nemotronh kimi; do
    checkpoint=$(model_field "$model" checkpoint)
    export_prefix=$(model_field "$model" prefix)
    if [[ $model == qwen3_5 ]]; then suffix=sr
    else suffix=asym_sr; fi
    if [[ $suite == main ]]; then
        for task in gsm8k_cot "${ruler[@]}"; do
            for wa in 16:16 8:16 8:8 4:8; do
                IFS=: read -r w a <<< "$wa"
                emit "w${w}a${a}_fp32" "$task" off value none "$w" "$a"
                emit "w${w}a${a}_int8" "$task" "int8_$suffix" value none "$w" "$a"
            done
        done
    else
        [[ $model != kimi ]] || continue
        case "$suite" in
            ranking)
                emit fp32 gsm8k_cot off value none 16 16
                for bits in 4 6 8; do
                    for axis in value key; do
                        emit "int${bits}_$axis" gsm8k_cot "int${bits}_$suffix" "$axis" none 16 16
                    done
                done ;;
            rotation)
                emit value gsm8k_cot int6 value none 16 16
                emit value_rk gsm8k_cot int6 value rk 16 16
                emit key gsm8k_cot int6 key none 16 16
                emit key_rv gsm8k_cot int6 key rv 16 16 ;;
            promotion)
                for task in gsm8k_cot gpqa_diamond_cot_ns aime; do
                    emit fp32 "$task" off value none 16 16
                    for bits in 4 6 8; do emit "int$bits" "$task" "int${bits}_$suffix" value none 16 16; done
                    while IFS=$'\t' read -r id mode rotation interval clipping parity headbits rest; do
                        [[ $id == "$model/INT"*to*value* ]] || continue
                        label=${id#*/}
                        [[ $model/$task/$label == $only ]] || continue
                        mode=${mode/_block_dk/}
                        mode=${mode/_block/}
                        file=$(mask_file "$headbits" "$id")
                        emit "$label" "$task" "$mode" value none 16 16 --headbits "$file"
                    done < "$repo/experiments/configs/delta_ell.tsv"
                done ;;
        esac
    fi
done
finish
