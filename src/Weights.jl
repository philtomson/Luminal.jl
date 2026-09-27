# Weight Loading Infrastructure for Luminal.jl
#
# Design:
#   A `WeightRegistry` accumulates a mapping of (safetensors_key -> graph_node_id)
#   as the model is constructed. The `load_weights!` function then reads a
#   safetensors file and injects the correct data into graph.tensors[node_id].
#
# Usage:
#   reg = WeightRegistry()
#   model = Whisper(graph, reg)        # builds graph, registers all weights
#   load_weights!(graph, reg, "openai/whisper-tiny")
#   exec_fn = compile(graph)

using SafeTensors

export WeightRegistry, register_weight!, tie_weight!, load_weights!, load_weights_hf!, load_weights_to_dict

# ──────────────────────────────────────────────────────────────────────────────
# WeightRegistry
# ──────────────────────────────────────────────────────────────────────────────

"""
    WeightRegistry

Maps a safetensors tensor name (String) to a Luminal graph node ID (Int).
Build one alongside the model and pass it to `load_weights!`.
"""
mutable struct WeightRegistry
    mapping::Dict{String, Int}  # safetensors key -> graph node_id
    ties::Dict{String, String}  # name -> name it shares data with, when the file lacks it
    WeightRegistry() = new(Dict{String, Int}(), Dict{String, String}())
end

"""
    tie_weight!(reg, name, source)

Tied weights: when the checkpoint has no tensor `name`, its node gets `source`'s
data (the same array, no copy). The tied tensor stays a separate graph node, so
e.g. a tied output head is still a matmul-only weight that `compile` can store
in reduced precision, while the embedding keeps reading the Float32 original.
"""
tie_weight!(reg::WeightRegistry, name::String, source::String) = (reg.ties[name] = source; reg)

# Fill tied nodes the checkpoint had no tensor for from their source's data
function _apply_ties!(graph, reg)
    for (name, src) in reg.ties
        id = get(reg.mapping, name, 0); sid = get(reg.mapping, src, 0)
        (id == 0 || sid == 0 || haskey(graph.tensors, (id, 1)) || !haskey(graph.tensors, (sid, 1))) && continue
        graph.tensors[(id, 1)] = graph.tensors[(sid, 1)]
    end
end

"""
    register_weight!(reg, name, t)

Record that the graph node for `t` should be filled with the safetensors
tensor named `name`.
"""
function register_weight!(reg::WeightRegistry, name::String, t::Luminal.GraphTensor)
    reg.mapping[name] = t.id
    return t
end

# ──────────────────────────────────────────────────────────────────────────────
# Weight Loading
# ──────────────────────────────────────────────────────────────────────────────

"""
    load_weights!(graph, reg, path; device=get_device())

Load all weights from a safetensors file (or directory of shards) into `graph`.

- `path`: path to a `.safetensors` file OR a directory containing `*.safetensors`
  shards (e.g. a HuggingFace model directory).
- `device`: target device; weights are converted to Float32 and moved onto it.

Only tensors registered in `reg` are loaded; additional tensors in the file
are silently ignored.
"""
function load_weights!(graph::Luminal.Graph,
                       reg::WeightRegistry,
                       path::String;
                       device::Luminal.AbstractDevice=Luminal.get_device())

    files = _collect_safetensors_files(path)
    isempty(files) && error("No safetensors files found at: $path")

    # Build a reverse map: key -> node_id for fast lookup
    key_to_node = reg.mapping   # String -> Int

    loaded = 0
    for file in files
        open(file, "r") do fio
            header, header_length = SafeTensors.load_header(fio)
            # Only iterate over keys we actually need
            for (key, node_id) in key_to_node
                sym = Symbol(key)
                !haskey(header, sym) && continue

                data = _load_tensor(fio, header[sym], header_length, device)
                data === nothing && error("Unsupported dtype $(header[sym][:dtype]) for tensor $key")

                graph.tensors[(node_id, 1)] = data
                loaded += 1
            end
        end
    end

    _apply_ties!(graph, reg)
    loaded += count(n -> haskey(reg.mapping, n) && haskey(graph.tensors, (reg.mapping[n], 1)), keys(reg.ties))
    n_params = length(reg.mapping)
    @info "Loaded $loaded/$n_params weights" path=path
    if loaded < n_params
        missing_keys = [k for (k, id) in reg.mapping
                        if !haskey(graph.tensors, (id, 1))]
        @warn "Missing weights" missing_keys
    end

    return graph
end

"""
    load_weights!(graph, reg, tensors; device=get_device())

Populate `graph` tensors from a pre-loaded dictionary of arrays/tensors.
If the tensors are already on the correct device, they are shared.
"""
function load_weights!(graph::Luminal.Graph,
                       reg::WeightRegistry,
                       tensors::Dict{String, Any};
                       device::Luminal.AbstractDevice=Luminal.get_device())
    loaded = 0
    for (i, (key, node_id)) in enumerate(reg.mapping)
        if haskey(tensors, key)
            data = tensors[key]
            target_data = Luminal.to_device(data, device)
            graph.tensors[(node_id, 1)] = target_data
            loaded += 1
        end
    end
    _apply_ties!(graph, reg)
    return graph
end

# Backward compatibility or simpler Dict type
function load_weights!(graph::Luminal.Graph, reg::WeightRegistry, tensors::Dict{String, <:AbstractArray}; device=get_device())
    return load_weights!(graph, reg, Dict{String, Any}(k => v for (k,v) in tensors); device=device)
end

# Read one safetensors entry as a Float32 array on `device`, in Julia's
# column-major layout with the tensor's logical (row-major) shape. The raw
# F32/F16/BF16 data is uploaded as stored (BF16 and F16 are half the bytes of
# Float32), then converted and transposed on the device. `nothing` for other dtypes.
function _load_tensor(fio::IO, entry, header_length::Integer, device::Luminal.AbstractDevice)
    dtype = String(entry[:dtype])
    T = dtype == "F32" ? Float32 : dtype == "F16" ? Float16 : dtype == "BF16" ? UInt16 : nothing
    T === nothing && return nothing
    shape = Int.(entry[:shape])
    start = Int(entry[:data_offsets][1]) + header_length
    stop  = Int(entry[:data_offsets][2]) + header_length
    seek(fio, start)
    raw = Vector{T}(undef, (stop - start) ÷ sizeof(T))
    read!(fio, raw)
    # Row-major data read column-major is the transpose: reversed dims.
    d = Luminal.to_device(Base.reshape(raw, reverse(shape)...), device)
    f = T === UInt16 ? reinterpret.(Float32, UInt32.(d) .<< 16) :   # BF16: the high half of a Float32
        T === Float16 ? Float32.(d) : d
    return length(shape) > 1 ? permutedims(f, length(shape):-1:1) : f
end

"""
    load_weights_to_dict(path; device=get_device()) -> Dict{String, Any}

Load all weights from a safetensors file (or directory) into a dictionary of
device-resident arrays. This is useful for sharing weights across multiple
graphs (e.g., prefill vs. decode).
"""
function load_weights_to_dict(path::String;
                             device::Luminal.AbstractDevice=Luminal.get_device())
    files = _collect_safetensors_files(path)
    isempty(files) && error("No safetensors files found at: $path")

    tensors = Dict{String, Any}()
    for file in files
        open(file, "r") do fio
            header, header_length = SafeTensors.load_header(fio)
            # We don't know which keys we need, so we load everything in the file(s)
            # that we recognize as weights.
            for (sym, entry) in header
                key = String(sym)
                key == "__metadata__" && continue
                haskey(tensors, key) && continue # Skip if already loaded from previous shard

                !haskey(entry, :dtype) && continue # Skip if not a tensor entry
                data = _load_tensor(fio, entry, header_length, device)
                data === nothing && continue       # unsupported dtype
                tensors[key] = data
            end
        end
    end
    return tensors
end

"""
    load_weights_hf!(graph, reg, model_id; cache_dir=nothing, device=get_device())

High-level helper: downloads (if needed) a HuggingFace model by `model_id`
and loads its safetensors weights.

Requires `huggingface-cli` or a manual download. If the directory already
exists, no download is performed.

Example:
    load_weights_hf!(graph, reg, "openai/whisper-tiny")
"""
function load_weights_hf!(graph::Luminal.Graph,
                          reg::WeightRegistry,
                          model_id::String;
                          cache_dir::Union{String, Nothing}=nothing,
                          device::Luminal.AbstractDevice=Luminal.get_device())

    dir = if cache_dir !== nothing
        cache_dir
    else
        # Default: ~/.cache/huggingface/hub/models--<org>--<model>
        home = get(ENV, "HOME", "/root")
        safe_id = replace(model_id, "/" => "--")
        joinpath(home, ".cache", "huggingface", "hub", "models--$(safe_id)", "snapshots")
    end

    # Check if already downloaded
    if !isdir(dir) || isempty(readdir(dir))
        @info "Downloading $model_id from HuggingFace..." dir=dir
        cmd = `huggingface-cli download $model_id --local-dir $dir`
        run(cmd)
    else
        # Use the most-recently modified snapshot
        snapshots = sort(readdir(dir; join=true), by=mtime)
        dir = last(snapshots)
        @info "Using cached model" dir=dir
    end

    return load_weights!(graph, reg, dir; device=device)
end

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

function _collect_safetensors_files(path::String)
    if isfile(path) && endswith(path, ".safetensors")
        return [path]
    elseif isdir(path)
        files = filter(f -> endswith(f, ".safetensors"), readdir(path; join=true))
        # Sort so shards are processed in order (model-00001-of-00002.safetensors, …)
        return sort(files)
    else
        return String[]
    end
end
