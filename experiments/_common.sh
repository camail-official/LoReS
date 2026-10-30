#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
python="$repo/.venv/bin/python"
output_root="$repo/output"
models_root="$repo/models"
execute=false
only='*'
args=()
while (($#)); do
    case "$1" in
        --python) python=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --models-root) models_root=$2; shift 2 ;;
        --run) execute=true; shift ;;
        --dry-run) execute=false; shift ;;
        --only) only=$2; shift 2 ;;
        --help|-h)
            printf '%s\n' "$usage" \
                'Common flags: --python PATH --models-root DIR --output-root DIR' \
                '              --only GLOB --run (default: print commands only)'
            exit 0 ;;
        *) args+=("$1"); shift ;;
    esac
done
set -- "${args[@]}"
matched=0

job() {
    local id=$1
    shift
    [[ $id == $only ]] || return 0
    matched=$((matched + 1))
    printf '# %s\n' "$id"
    printf '%q ' "$@"
    printf '\n'
    if "$execute"; then "$@"; fi
}

finish() {
    ((matched)) || { printf 'No configurations match --only %s\n' "$only" >&2; exit 2; }
}

model_field() {
    "$python" -c 'import sys; from lores.models.registry import model_info; m = model_info(sys.argv[1]); print(m.get(sys.argv[2], m["checkpoint"]))' "$1" "$2"
}

checkpoint_pin() {
    "$python" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' \
        "$repo/experiments/configs/checkpoints.json" "$1" "$2"
}

# mask_file STEM CONFIG_ID: a mask of the allocation run passed with --masks, checked against the row's bit widths.
mask_file() {
    local stem=$1 id=$2
    if [[ -z $masks ]]; then
        if "$execute"; then printf 'Pass the matching allocation run of calibration.sh with --masks.\n' >&2; return 1; fi
        printf '<allocation-run>/%s.json' "$stem"
        return
    fi
    "$python" - "$masks/allocation.json" "$id" <<'EOF' || return 1
import json, re, sys
a, cid = json.load(open(sys.argv[1])), sys.argv[2]
model, label = cid.split("/", 1)
m = re.match(r"INT(\d)to(\d)-(\d+)pct", label)
scope = "global" if not m and model in ("qwen3_5", "nemotronh", "kimi") else "per-layer"
want = (model, *((int(m[1]), int(m[2]), int(m[3]) / 100) if m else (4, 6, 0.5)), scope)
got = (a["model"], a["base_bits"], a["promoted_bits"], a["fraction"], a["budget_scope"])
if got != want:
    sys.exit(f"{cid} needs an allocation with (model, base, promoted, fraction, scope) = {want}, got {got}")
EOF
    printf '%s' "$masks/$stem.json"
}

zoo_cells=$("$python" -c 'from lores.models.registry import MIXER_ZOO_CELLS; print(" ".join(MIXER_ZOO_CELLS))')
read -r -a zoo <<< "$zoo_cells"
