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

export WeightRegistry, register_weight!, load_weights!, load_weights_hf!, load_weights_to_dict

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

                entry   = header[sym]
                dtype   = String(entry[:dtype])
                shape   = tuple(Int.(entry[:shape])...)
                start   = Int(entry[:data_offsets][1]) + header_length
                stop    = Int(entry[:data_offsets][2]) + header_length

                # Read raw bytes directly to bypass SafeTensors.jl's BF16 error
                seek(fio, start)
                raw_bytes = read(fio, stop - start)

                # Convert based on dtype
                if dtype == "F32"
                    data = reinterpret(Float32, raw_bytes)
                elseif dtype == "BF16"
                    # Reinterpret as UInt16, cast to UInt32, shift left 16, reinterpret as Float32
                    u16 = reinterpret(UInt16, raw_bytes)
                    data = map(u -> reinterpret(Float32, UInt32(u) << 16), u16)
                elseif dtype == "F16"
                    data = Float32.(reinterpret(Float16, raw_bytes))
                else
                    error("Unsupported dtype $dtype for tensor $key")
                end

                # Reshape and permute to match Julia's column-major format
                data = Base.reshape(collect(data), reverse(shape)...)
                if length(shape) > 1
                    data = Array(permutedims(data, length(shape):-1:1))
                end

                graph.tensors[(node_id, 1)] =
                    Luminal.to_device(Dict(node_id => data), device)[node_id]

                # Let GC collect the raw array
                raw_bytes = nothing
                data = nothing
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
    return graph
end

# Backward compatibility or simpler Dict type
function load_weights!(graph::Luminal.Graph, reg::WeightRegistry, tensors::Dict{String, <:AbstractArray}; device=get_device())
    return load_weights!(graph, reg, Dict{String, Any}(k => v for (k,v) in tensors); device=device)
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
                dtype   = String(entry[:dtype])
                shape   = tuple(Int.(entry[:shape])...)
                offsets = entry[:data_offsets]
                start   = Int(offsets[1]) + header_length
                stop    = Int(offsets[2]) + header_length

                seek(fio, start)
                raw_bytes = read(fio, stop - start)

                if dtype == "F32"
                    data = reinterpret(Float32, raw_bytes)
                elseif dtype == "BF16"
                    u16 = reinterpret(UInt16, raw_bytes)
                    data = map(u -> reinterpret(Float32, UInt32(u) << 16), u16)
                elseif dtype == "F16"
                    data = reinterpret(Float16, raw_bytes)
                else
                    # Skip unknown dtypes (like metadata)
                    continue
                end

                # Reshape and permute to match Julia's column-major format
                permuted = Base.reshape(collect(data), reverse(shape)...)
                if length(shape) > 1
                    permuted = Array(permutedims(permuted, length(shape):-1:1))
                end
                
                # Convert to Float16 if on GPU to save memory
                if device isa Luminal.CUDADevice || device isa Luminal.AMDDevice
                    permuted = convert(Array{Float16}, permuted)
                else
                    permuted = convert(Array{Float32}, permuted)
                end

                tensors[key] = Luminal.to_device(permuted, device)
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
