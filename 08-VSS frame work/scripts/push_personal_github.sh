#!/bin/sh
# Publish only the VSS source directory to Tahereh Fahi's personal MNL main.
# Generated runs, inputs, and third-party assets are excluded by .gitignore.
set -eu

framework_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$framework_dir/.." && pwd)
vss_path='08-VSS frame work'
expected_remote='https://github.com/tahereh-fahi/MNL.git'

if [ "$(git -C "$repo_dir" rev-parse --show-toplevel)" != "$repo_dir" ]; then
    printf '%s\n' 'Refusing: unexpected Git repository.' >&2
    exit 1
fi
if [ "$(git -C "$repo_dir" branch --show-current)" != 'main' ]; then
    printf '%s\n' 'Refusing: check out the MNL main branch first.' >&2
    exit 1
fi
if [ "$(git -C "$repo_dir" remote get-url origin)" != "$expected_remote" ]; then
    printf '%s\n' 'Refusing: origin is not the expected personal MNL repository.' >&2
    exit 1
fi

git -C "$repo_dir" fetch origin main
if ! git -C "$repo_dir" merge-base --is-ancestor origin/main HEAD; then
    printf '%s\n' 'Refusing: local main is not based on the current GitHub main.' >&2
    exit 1
fi

# A failed previous push may leave VSS-only commits ahead. Never carry an
# unrelated local commit into the next VSS push without separate review.
other_paths=$(git -C "$repo_dir" log --format= --name-only origin/main..HEAD \
    | sed '/^$/d' | grep -v '^08-VSS frame work/' || true)
if [ -n "$other_paths" ]; then
    printf '%s\n' 'Refusing: unpushed commits also change files outside VSS:' >&2
    printf '%s\n' "$other_paths" >&2
    exit 1
fi

git -C "$repo_dir" add -A -- "$vss_path"

forbidden=$(git -C "$repo_dir" ls-files --cached -- "$vss_path" \
    | grep -E '(^|/)(runs|inputs|assets|fixtures|\.venv|\.cache)(/|$)|\.(png|jpe?g|mp4|mov|avi|mkv|pt|pth|onnx|zip|pdf)$' || true)
if [ -n "$forbidden" ]; then
    printf '%s\n' 'Refusing: a generated or restricted VSS file is tracked:' >&2
    printf '%s\n' "$forbidden" >&2
    exit 1
fi

if ! git -C "$repo_dir" diff --cached --quiet -- "$vss_path"; then
    printf '\n%s\n' 'VSS changes proposed for the public repository:'
    git -C "$repo_dir" diff --cached --stat -- "$vss_path"
    git -C "$repo_dir" diff --cached --name-only -- "$vss_path"
    printf '\n%s' 'Type PUSH to commit these VSS files and push main: '
    read -r answer
    if [ "$answer" != 'PUSH' ]; then
        printf '%s\n' 'Cancelled. No commit or push was made.'
        exit 0
    fi
    git -C "$repo_dir" commit --only -m "Update VSS source $(date +%F)" -- "$vss_path"
fi

git -C "$repo_dir" push origin main
printf '%s\n' 'VSS source push completed.'
