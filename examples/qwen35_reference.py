"""Reference data for Qwen3.5 / Qwen3.6 (`qwen3_5`) from a tiny random model.

Usage: python3 examples/qwen35_reference.py <out_dir> [seed]

Builds a tiny `Qwen3_5ForCausalLM` with transformers' own classes (3 Gated
DeltaNet layers and 1 gated full-attention layer; unequal key / value head
sizes so layout mistakes can't cancel; every norm, gate and decay parameter
randomized), saves it (`config.json`, `model.safetensors`), and records, in
Float32 on the CPU:
- a batch-2 prefill of 7 tokens, then 3 greedy decode steps with the cache;
- layer 0's Gated DeltaNet input and output for each of those calls
  (`dn_in_*`, `dn_out_*`) and its states after each (`conv_*`: the last
  kernel-size pre-convolution inputs, `rec_*`: the recurrent state);
- the model's logits for each call (`logits_*`).
Arrays are raw little-endian float32 in C order; `index.json` lists names and
shapes (a C-order (a, b, c) array reads as a column-major Julia (c, b, a)).
"""
import json, os, sys
import torch
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
os.makedirs(out, exist_ok=True)
torch.manual_seed(seed)

cfg = Qwen3_5TextConfig(
    vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=12,
    linear_conv_kernel_dim=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_interleaved": True, "mrope_section": [1, 1, 0]},
    rms_norm_eps=1e-6, tie_word_embeddings=False, hidden_act="silu",
)
model = Qwen3_5ForCausalLM(cfg).float().eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):
            p.copy_(0.2 * torch.randn_like(p))              # (1 + w) norms, and the gated norm's w
        elif name.endswith("dt_bias"):
            p.copy_(torch.randn_like(p))
        elif name.endswith("A_log"):
            p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4)))
        else:
            p.copy_(0.3 * torch.randn_like(p))
model.save_pretrained(out, safe_serialization=True)

arrays = {}
def put(name, t):
    t = t.detach().float().contiguous()
    t.numpy().tofile(os.path.join(out, name + ".f32"))
    arrays[name] = list(t.shape)

dn = model.model.layers[0].linear_attn
cap = {}
dn.register_forward_hook(lambda m, args, kwargs, o: cap.update(x=kwargs["hidden_states"], y=o), with_kwargs=True)

ids = torch.randint(0, cfg.vocab_size, (2, 7))
steps = []
with torch.no_grad():
    o = model(ids, use_cache=True)
    cache = o.past_key_values
    put("ids", ids.float()); put("logits_0", o.logits)
    put("dn_in_0", cap["x"]); put("dn_out_0", cap["y"])
    put("conv_0", cache.layers[0].conv_states); put("rec_0", cache.layers[0].recurrent_states)
    nxt = o.logits[:, -1].argmax(-1, keepdim=True)
    for s in range(1, 4):
        steps.append(nxt[:, 0].tolist())
        o = model(nxt, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        put(f"logits_{s}", o.logits)
        put(f"dn_in_{s}", cap["x"]); put(f"dn_out_{s}", cap["y"])
        put(f"conv_{s}", cache.layers[0].conv_states); put(f"rec_{s}", cache.layers[0].recurrent_states)
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)

json.dump({"arrays": arrays, "decode_tokens": steps}, open(os.path.join(out, "index.json"), "w"), indent=1)
print("saved", out, {k: v for k, v in arrays.items() if k.endswith("_0")}, "decode tokens", steps)
