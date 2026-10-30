#!/usr/bin/env bash
usage='prompts.sh --directory NEW_DIRECTORY [--cache-dir DIR]'
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
directory= cache=
while (($#)); do
    case "$1" in
        --directory) directory=$2; shift 2 ;;
        --cache-dir) cache=$2; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ $directory == /* ]] || directory="$repo/$directory"
[[ $directory == "$output_root/new/"* ]] || { printf 'Choose a fresh --directory under %s/new/\n' "$output_root" >&2; exit 2; }
if "$execute"; then
    "$python" -c 'import sys; from lores.evaluation.output_store import writable_output; writable_output(sys.argv[1], root=sys.argv[2]).mkdir(parents=True, exist_ok=False)' "$directory" "$output_root"
fi
for task in aime gpqa mmlupro; do
    extra=()
    [[ $task != aime ]] || extra+=(ns)
    [[ -z $cache ]] || extra+=(--cache-dir "$cache")
    job "$task" "$python" -m "scripts.evaluation.prompts.$task" "$directory/$task.jsonl" "${extra[@]}"
done
finish
