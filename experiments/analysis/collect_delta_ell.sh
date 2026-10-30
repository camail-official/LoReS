#!/usr/bin/env bash
usage='collect_delta_ell.sh --selection FILE.tsv
Each row is CONFIG_ID<TAB>EXACT_RUN_DIRECTORY, using IDs from experiments/configs/delta_ell.tsv.'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
[[ $# == 2 && $1 == --selection ]] || { printf '%s\n' "$usage" >&2; exit 2; }
job collect "$python" -m scripts.evaluation.verify_delta_ell --selection "$2" --sequences 8 --seqlen 2048 \
    --save-report --output-root "$output_root"
finish
