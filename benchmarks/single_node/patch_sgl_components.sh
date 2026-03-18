#!/usr/bin/env bash
# Reinstall aiter and sglang from custom remote/ref.
# Expects AITER_REMOTE, AITER_REF, SGLANG_REMOTE, SGLANG_REF from env.
# If no refs are provided, skip entirely and use default packages in container.

set -e

if [[ -z "$AITER_REF" && -z "$SGLANG_REF" ]]; then
    echo "No patch refs provided; using default packages in container."
    exit 0
fi

work_dir="/sgl-workspace"
aiter_remote="${AITER_REMOTE:-https://github.com/zhentaocc/aiter.git}"
aiter_ref="${AITER_REF:-mi35_qwen35_image}"
sglang_remote="${SGLANG_REMOTE:-https://github.com/sgl-project/sglang}"
sglang_ref="${SGLANG_REF:-mi35_qwen35_image}"

if [[ ! -d "$work_dir" ]]; then
    echo "$work_dir not found; assuming image ships correct versions."
    exit 0
fi

if [[ -n "$AITER_REF" ]]; then
    pip uninstall amd-aiter -y
    cd "$work_dir/aiter"
    git remote set-url origin "$aiter_remote" 2>/dev/null || git remote add origin "$aiter_remote"
    git fetch origin "$aiter_ref" 2>/dev/null || git fetch origin
    git checkout "$aiter_ref" 2>/dev/null || git reset --hard "origin/$aiter_ref" 2>/dev/null || git reset --hard "$aiter_ref"
    rm -rf aiter/jit/*.so 2>/dev/null || true
    rm -rf aiter/jit/build 2>/dev/null || true
    rm -rf aiter/jit/dist 2>/dev/null || true
    PREBUILD_KERNELS=0 python setup.py develop
    echo "aiter ($aiter_ref) installed from $aiter_remote"
else
    echo "AITER_REF not set; using default aiter in container"
fi

if [[ -n "$SGLANG_REF" ]] && [[ -d "$work_dir/sglang/.git" ]]; then
    cd "$work_dir/sglang"
    git remote set-url origin "$sglang_remote" 2>/dev/null || git remote add origin "$sglang_remote"
    git fetch origin "$sglang_ref" 2>/dev/null || git fetch origin
    git checkout "$sglang_ref" 2>/dev/null || git reset --hard "origin/$sglang_ref" 2>/dev/null || git reset --hard "$sglang_ref"
    echo "sglang ($sglang_ref) from $sglang_remote"
elif [[ -z "$SGLANG_REF" ]]; then
    echo "SGLANG_REF not set; using default sglang in container"
fi
