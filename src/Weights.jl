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

export WeightRegistry, register_weight!, load_weights!, load_weights_hf!

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
    WeightRegistry() = new(Dict{String, Int}())
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

    # Collect all safetensors files.
    files = _collect_safetensors_files(path)
    isempty(files) && error("No safetensors files found at: $path")

    loaded = 0
    for file in files
        tensors = SafeTensors.load_safetensors(file)   # Dict{String, Array}
        for (key, node_id) in reg.mapping
            if haskey(tensors, key)
                raw = tensors[key]
                # Convert to Float32; SafeTensors.jl already permutes dims for Julia
                data = convert(Array{Float32}, raw)
                # Move to the target device (CuArray, etc.)
                graph.tensors[(node_id, 1)] = Luminal.to_device(Dict(node_id => data), device)[node_id]
                loaded += 1
            end
        end
    end

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

Populate `graph` tensors from a pre-loaded dictionary of arrays (e.g. from `SafeTensors.load_safetensors`).
"""
function load_weights!(graph::Luminal.Graph,
                       reg::WeightRegistry,
                       tensors::Dict{String, <:AbstractArray};
                       device::Luminal.AbstractDevice=Luminal.get_device())
    loaded = 0
    for (key, node_id) in reg.mapping
        if haskey(tensors, key)
            raw = tensors[key]
            data = convert(Array{Float32}, raw)
            graph.tensors[(node_id, 1)] = Luminal.to_device(Dict(node_id => data), device)[node_id]
            loaded += 1
        end
    end
    return graph
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
