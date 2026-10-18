"""Convert a released NCM checkpoint (hazyresearch/ncm-checkpoints) to safetensors for Luminal.

    python examples/ncm_convert.py ~/devel/AI/ncm_artifacts/tinystories

Reads generator.pt and tokenizer.pt with torch.load(weights_only=True) (no pickled code runs) and writes
generator.safetensors and tokenizer.safetensors next to them. Training-only state (EMA statistics, hit counters,
the precomputed attention mask) is dropped; the level-0 unigram table that seeds generation is kept as `level0_unigram`.
"""
import sys
import torch
from safetensors.torch import save_file

DROP = ("ema_cluster_size_", "ema_w_", "codebook_hit_", "attn_mask_for_masking", "lvl_1L")

def convert(src, dst, pick):
    ck = torch.load(src, map_location="cpu", weights_only=True)
    sd = {k: v.contiguous() for k, v in ck["model"].items() if not any(d in k for d in DROP)}
    sd.update(pick(ck))
    save_file(sd, dst)
    print(f"{dst}: {len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.1f}M values")

if __name__ == "__main__":
    d = sys.argv[1]
    convert(f"{d}/generator.pt", f"{d}/generator.safetensors",
            lambda ck: {"level0_unigram": ck["inference"]["level0_unigram"].float().contiguous()})
    convert(f"{d}/tokenizer.pt", f"{d}/tokenizer.safetensors", lambda ck: {})
