"""Dump Hugging Face reference outputs for a Llama-family checkpoint.

Usage: python3 examples/llama_reference.py <model_dir> <out_dir> [max_new_tokens]

For each chat prompt below (formatted with the tokenizer's chat template), writes:
- the prompt token ids and HF's greedy continuation (argmax, no sampling or
  logit processors) to index.json, with the decoded text;
- the Float32 logits at every prompt position as raw little-endian float32
  (C order (S, V), which reads as a column-major Julia (V, S) array).
The model runs in Float32 on the CPU.
"""
import json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, out = sys.argv[1], sys.argv[2]
max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 40
os.makedirs(out, exist_ok=True)
PROMPTS = ["What is the capital of France? Answer in one sentence.",
           "Write a haiku about a lighthouse.",
           "Explain in two sentences why the sky is blue."]

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
gen = json.load(open(os.path.join(model_dir, "generation_config.json")))
eos = gen["eos_token_id"]
eos = set(eos if isinstance(eos, list) else [eos])

index = {"prompts": []}
with torch.no_grad():
    for i, p in enumerate(PROMPTS):
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
        out_ = model(torch.tensor([ids]), use_cache=True)
        logits = out_.logits[0]                                   # (S, V)
        logits.float().numpy().tofile(os.path.join(out, f"logits_{i}.f32"))
        past, nxt, gen_ids = out_.past_key_values, int(logits[-1].argmax()), []
        while True:
            gen_ids.append(nxt)
            if nxt in eos or len(gen_ids) >= max_new:
                break
            o = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past, nxt = o.past_key_values, int(o.logits[0, -1].argmax())
        index["prompts"].append({"prompt": p, "ids": ids, "logits_shape": list(logits.shape),
                                 "generated": gen_ids,
                                 "text": tok.decode(gen_ids, skip_special_tokens=True)})
        print(p, "->", repr(index["prompts"][-1]["text"]), flush=True)

json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
