# lsa_pt_to_safetensors.py -- convert a latent-sparse-attention map file (a torch .pt of fp32 matrices) to safetensors, with no torch.
#
#   python3 -I examples/lsa_pt_to_safetensors.py qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt qwen38_27b__...-d2048.safetensors
#
# The D-LSA maps are released at https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/d-lsa-weights-v1. A .pt is a zip of raw
# storages and a pickle; this reads the pickle with a restricted unpickler that rebuilds tensor metadata and refuses every other class, so no
# code in the file runs, then copies the fp32 storages into a safetensors file. `Luminal.lsa_load_maps` reads the result.

import collections, json, pickle, struct, sys, zipfile

class Tensor:
    def __init__(self, key, offset, size, stride): self.key, self.offset, self.size, self.stride = key, offset, tuple(size), tuple(stride)
def rebuild(storage, offset, size, stride, *rest): return Tensor(storage[0], offset, size, stride)
class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"): return rebuild
        if (module, name) == ("collections", "OrderedDict"): return collections.OrderedDict
        if module == "torch" and name.endswith("Storage"): return name
        raise pickle.UnpicklingError(f"blocked {module}.{name}")
    def persistent_load(self, pid):
        # ('storage', StorageType, key, location, numel)
        assert pid[0] == "storage" and pid[1] == "FloatStorage", pid
        return (pid[2], pid[4])

src, dst = sys.argv[1:3]
z = zipfile.ZipFile(src)
root = z.namelist()[0].split("/")[0]
tensors = Unpickler(z.open(f"{root}/data.pkl")).load()
header, blobs, position = {}, [], 0
for name, t in tensors.items():
    rows, cols = t.size
    assert t.offset == 0 and t.stride == (cols, 1), (name, t.offset, t.stride)
    data = z.read(f"{root}/data/{t.key}")[: rows * cols * 4]
    assert len(data) == rows * cols * 4
    header[name] = {"dtype": "F32", "shape": [rows, cols], "data_offsets": [position, position + len(data)]}
    blobs.append(data); position += len(data)
header["__metadata__"] = {"format": "pt"}
raw = json.dumps(header, separators=(",", ":")).encode()
raw += b" " * (-len(raw) % 8)
with open(dst, "wb") as out:
    out.write(struct.pack("<Q", len(raw))); out.write(raw)
    for b in blobs: out.write(b)
print(dst.split("/")[-1], len(tensors), "tensors", position, "bytes")
