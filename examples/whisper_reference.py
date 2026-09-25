"""Dump Hugging Face whisper reference tensors for tests/validation.

Usage: python3 examples/whisper_reference.py <model_dir> <audio.f32> <out_dir>

<audio.f32> is raw 16 kHz mono float32, e.g. for the clip the tests use:
  espeak-ng -s 150 -w clip.wav "The quick brown fox jumps over the lazy dog. Whisper is running in Julia."
  ffmpeg -i clip.wav -f f32le -ac 1 -ar 16000 audio.f32

Every tensor is written as raw little-endian float32 in C order, with its
shape in index.json. A C-order (B, S, D) tensor reads directly as a
column-major Julia (D, S, B) array. Encoder/decoder outputs are dumped twice:
with the exact erf GELU (`*.exact`) and with the tanh approximation Luminal
uses (`*.tanh`), so layout errors can be separated from activation error.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

model_dir, audio_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
index = {}

def dump(name, t):
    a = t.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(t) else np.asarray(t, np.float32)
    a = np.ascontiguousarray(a)
    a.tofile(os.path.join(out, name + ".f32"))
    index[name] = list(a.shape)

audio = np.fromfile(audio_path, dtype=np.float32)
fe = WhisperFeatureExtractor.from_pretrained(model_dir)
dump("mel_filters", fe.mel_filters)                        # (201, 80)
feats = fe(audio, sampling_rate=16000, return_tensors="pt").input_features  # (1, 80, 3000)
dump("mel", feats)

tok = WhisperTokenizer.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
prompt = [tok.convert_tokens_to_ids(t) for t in
          ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]]
index["prompt"] = prompt

def run(tag):
    with torch.no_grad():
        enc = model.model.encoder(feats, output_hidden_states=True)
        dump(f"enc_embed.{tag}", enc.hidden_states[0])          # after conv + positions
        dump(f"enc_layer0.{tag}", enc.hidden_states[1])
        dump(f"enc_out.{tag}", enc.last_hidden_state)            # after final layer norm
        # Greedy loop without generate()'s logit processors.
        ids = list(prompt)
        for _ in range(60):
            logits = model(encoder_outputs=(enc.last_hidden_state,),
                           decoder_input_ids=torch.tensor([ids])).logits
            nxt = int(logits[0, -1].argmax())
            ids.append(nxt)
            if nxt == tok.eos_token_id:
                break
        logits = model(encoder_outputs=(enc.last_hidden_state,),
                       decoder_input_ids=torch.tensor([ids[:-1]])).logits
        dump(f"logits.{tag}", logits)                              # (1, S, V)
        index[f"tokens.{tag}"] = ids
        index[f"text.{tag}"] = tok.decode(ids, skip_special_tokens=True)

run("exact")
for m in model.modules():
    if hasattr(m, "activation_fn"):
        m.activation_fn = torch.nn.GELU(approximate="tanh")
run("tanh")

with open(os.path.join(out, "index.json"), "w") as f:
    json.dump(index, f, indent=1)
print(json.dumps({k: v for k, v in index.items() if not k.startswith("logits")}, indent=1)[:2000])
