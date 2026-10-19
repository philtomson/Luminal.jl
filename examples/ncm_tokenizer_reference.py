"""PyTorch reference for the Luminal NCM tokenizer side: the BPE ids the repo's quantizer and decoder produce for given codes.

    python examples/ncm_tokenizer_reference.py <ncm repo> <checkpoint dir> <codes.txt>

`codes.txt`: one line per level, the level's codes (0-based) for sample 1, then the Julia-side decoded ids as the last line.
Prints how many of the 128 positions differ.
"""
import sys, types
from unittest import mock
import torch, yaml

repo, ckpt, codes_file = sys.argv[1:4]
sys.path.insert(0, repo)
for m in ("hydra", "omegaconf"):
    sys.modules[m] = mock.MagicMock()
from models.quant import MultiscaleResidualVectorQuantizer   # noqa: E402
from models.vae import Decoder                                # noqa: E402

cfg = yaml.safe_load(open(f"{ckpt}/config.yaml"))["tokenizer"]["vqvae"]
vqc = dict(cfg["vector_quantizer_config"])
levels = vqc["levels"]["manual_levels"]
tok = torch.load(f"{ckpt}/tokenizer.pt", map_location="cpu", weights_only=True)["model"]
sub = lambda p: {k[len(p):]: v for k, v in tok.items() if k.startswith(p)}
vq = MultiscaleResidualVectorQuantizer.from_config({
    "vqvae_vocab_size": cfg["vqvae_vocab_size"], "embed_dim": cfg["embed_dim"],
    **{k: v for k, v in vqc.items() if k != "clss"},
    "levels": types.SimpleNamespace(use_manual_levels=True, manual_levels=levels),
    "quant_resi": types.SimpleNamespace(**vqc["quant_resi"]),
    "aux": None})
vq.load_state_dict(sub("vq."), strict=False)
dec = Decoder(cfg["in_vocab_size"], cfg["embed_dim"], cfg["n_layers"], compression_factor=cfg["compression_factor"],
              dropout=0.0, use_qk_norm=cfg["use_qk_norm"], dec_layers=1, use_pos_emb=False, context_length=cfg["context_length"])
dec.load_state_dict(sub("decoder."), strict=False)
vq.eval(); dec.eval()

lines = [list(map(int, l.split())) for l in open(codes_file) if l.strip()]
idx = torch.tensor([sum(lines[:-1], [])])
jl_ids = torch.tensor(lines[-1])
with torch.no_grad():
    _, zl = vq.idx_to_zhats(idx, return_zlevels=True)
    ids = dec(zl[-1]).argmax(-1)[0]
print("positions", ids.numel(), "differing", int((ids != jl_ids).sum()))
