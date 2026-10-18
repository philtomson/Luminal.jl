"""PyTorch reference for the Luminal NCM generator: per-level logits on fixed random inputs.

    python examples/ncm_reference.py <ncm repo> <checkpoint dir> <out.safetensors>

Builds the generator's transformer stack and tied codebook heads from the NCM repo's own modules
(models.common.Block, models.next_concept._TiedCodebookHead) and runs the generation-time computation of
NextConceptTransformer._decode_level_logits for every level. hydra / omegaconf are only needed by the repo's
VQ-VAE, which is not used here, so they are stubbed.
"""
import sys, types
from unittest import mock
import torch, yaml
from safetensors.torch import save_file

repo, ckpt, out = sys.argv[1:4]
sys.path.insert(0, repo)
for m in ("hydra", "omegaconf"):
    sys.modules[m] = mock.MagicMock()
from models.common import Block, LayerNorm, precompute_rope_cos_sin   # noqa: E402
from models.next_concept import _TiedCodebookHead                      # noqa: E402

cfg = yaml.safe_load(open(f"{ckpt}/config.yaml"))["model"]
gen = torch.load(f"{ckpt}/generator.pt", map_location="cpu", weights_only=True)["model"]
tok = torch.load(f"{ckpt}/tokenizer.pt", map_location="cpu", weights_only=True)["model"]
levels = yaml.safe_load(open(f"{ckpt}/config.yaml"))["tokenizer"]["vqvae"]["vector_quantizer_config"]["levels"]["manual_levels"]
n_embd, n_head, P = cfg["n_embd"], cfg["n_head"], cfg["prefix_len"]
embed_dim = gen["transformer.wte.weight"].shape[1]
conf = types.SimpleNamespace(**cfg)

torch.manual_seed(0)
sub = lambda sd, p: {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
blocks = torch.nn.ModuleList([Block(conf) for _ in range(cfg["n_layer"])])
blocks.load_state_dict(sub(gen, "transformer.blocks."))
ln_f = LayerNorm(n_embd, bias=cfg["bias"]); ln_f.load_state_dict(sub(gen, "transformer.ln_f."))
wte = torch.nn.Linear(embed_dim, n_embd); wte.load_state_dict(sub(gen, "transformer.wte."))
wle = gen["transformer.wle.weight"]
heads = []
for i in range(len(levels) - 1):
    h = _TiedCodebookHead(n_embd, tok[f"vq.embeddings.{i + 1}.weight"], learn_scale=True, trainable=False, use_bias=False)
    h.load_state_dict(sub(gen, f"lm_heads.{i}."), strict=False)
    heads.append(h)
for m in (blocks, ln_f, wte, *heads):
    m.eval()

B = 2
prefix_tok = torch.tensor([[50256 % tok["encoder.wte.weight"].shape[0]]] * B)
res = {}
with torch.no_grad():
    prefix_raw = tok["encoder.wte.weight"][prefix_tok[:, :P]]
    res["prefix_raw"] = prefix_raw.contiguous()
    prefix_emb = wte(prefix_raw)
    for k in range(len(levels) - 1):
        L = levels[k + 1]
        x_in = torch.randn(B, L, embed_dim)
        x = torch.cat([prefix_emb, wte(x_in) + wle[k]], dim=1)
        mask = torch.zeros(1, 1, P + L, P + L)
        mask[:, :, :P, P:] = -torch.inf
        pos = torch.cat([torch.arange(P), torch.arange(L)])
        rope = precompute_rope_cos_sin(pos, n_embd // n_head, cfg["rope_base"])
        for b in blocks:
            x = b(x, attn_mask=mask, rope_cos_sin=rope)
        logits = heads[k](ln_f(x)[:, P:, :])
        res[f"x{k}"], res[f"logits{k}"] = x_in, logits.contiguous()
        print(f"level {k}: L={L} logits {tuple(logits.shape)} max|.|={logits.abs().max():.2f}")
save_file(res, out)
