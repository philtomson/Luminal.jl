#!/bin/bash
# Runs decode_attn_chunk.jl for DA_CH = 128, 256, 512: one git worktree per value (DA_CH is a compile-time constant; the score array of the
# kernel is DA_CH * 24 floats). Usage: benchmarks/decode_attn_chunk.sh <scratch dir>     (uses the GPU)
set -u
SCR=${1:?scratch dir}; ROOT=$(cd "$(dirname "$0")/.." && pwd); mkdir -p "$SCR"
for CH in 256 128 512; do
    WT=$SCR/wt_da$CH
    [ -d "$WT" ] || git -C "$ROOT" worktree add -q --detach "$WT" HEAD
    cp -n "$ROOT/Manifest.toml" "$WT/" 2>/dev/null
    sed -i "s/^const DA_CH = [0-9]*/const DA_CH = $CH/; s/(Float32, ([0-9]*,), false)      # (key, row)/(Float32, ($((CH * 24)),), false)      # (key, row)/" "$WT/src/DecodeAttn.jl"
    julia --project="$WT" "$ROOT/benchmarks/decode_attn_chunk.jl" "ch$CH" "$SCR/ref" 2>&1 | grep -v "^\s*$" | tail -12
done
