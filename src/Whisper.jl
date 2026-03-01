# Whisper Architecture definitions
# Includes a self-contained audio preprocessing pipeline (no librosa dependency)

using FFTW

# ─────────────────────────────────────────────────────────────────────────────
# Audio Preprocessing: Mel Spectrogram (matches OpenAI Whisper's audio.py)
# ─────────────────────────────────────────────────────────────────────────────

const SAMPLE_RATE  = 16000
const N_FFT        = 400
const HOP_LENGTH   = 160
const CHUNK_LENGTH = 30
const N_SAMPLES    = CHUNK_LENGTH * SAMPLE_RATE   # 480_000 samples in 30 s
const N_FRAMES     = N_SAMPLES ÷ HOP_LENGTH       # 3000 frames per spectrogram

# ── Mel Filterbank ─────────────────────────────────────────────────────────

"""
    hz_to_mel(hz)

Convert a frequency in Hz to the HTK Mel scale used by Whisper.
"""
hz_to_mel(hz::Real) = 2595.0 * log10(1.0 + hz / 700.0)

"""
    mel_to_hz(mel)

Convert a Mel-scale value back to Hz.
"""
mel_to_hz(mel::Real) = 700.0 * (10.0^(mel / 2595.0) - 1.0)

"""
    mel_filters(n_mels::Int=80; sr=SAMPLE_RATE, n_fft=N_FFT, fmin=0.0, fmax=sr/2) -> Matrix{Float32}

Build and return the (n_mels × n_fft÷2+1) triangular Mel filterbank matrix.
Replicates `librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)` exactly.
The result is cached after the first call.
"""
function mel_filters(n_mels::Int=80;
                     sr::Int=SAMPLE_RATE,
                     n_fft::Int=N_FFT,
                     fmin::Float64=0.0,
                     fmax::Float64=Float64(sr) / 2.0)

    n_freqs = n_fft ÷ 2 + 1                         # 201 bins

    # Linear frequency axis of the STFT bins
    fft_freqs = Float64[k * sr / n_fft for k in 0:(n_freqs - 1)]

    # Mel points: n_mels + 2 to get n_mels interior filters
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = [mel_to_hz(m) for m in range(mel_min, mel_max; length=n_mels + 2)]

    # Build the filterbank (n_mels × n_freqs)
    filters = zeros(Float32, n_mels, n_freqs)
    for m in 1:n_mels
        f_left   = mel_points[m]
        f_center = mel_points[m + 1]
        f_right  = mel_points[m + 2]
        for k in 1:n_freqs
            f = fft_freqs[k]
            if f_left <= f <= f_center
                filters[m, k] = Float32((f - f_left) / (f_center - f_left))
            elseif f_center < f <= f_right
                filters[m, k] = Float32((f_right - f) / (f_right - f_center))
            end
        end
    end

    # Slaney-style normalization: divide each filter by its width in Hz
    # (librosa uses norm="slaney" by default, which makes filters unit-area)
    for m in 1:n_mels
        width_hz = mel_points[m + 2] - mel_points[m]
        if width_hz > 0
            filters[m, :] ./= Float32(width_hz)
        end
    end

    return filters
end

# Pre-compute and cache the filterbank
const MEL_FILTERS_80  = mel_filters(80)
const MEL_FILTERS_128 = mel_filters(128)

"""
    get_mel_filters(n_mels) -> Matrix{Float32}

Return the cached Mel filterbank for 80 or 128 bins.
"""
function get_mel_filters(n_mels::Int)
    n_mels == 80  && return MEL_FILTERS_80
    n_mels == 128 && return MEL_FILTERS_128
    error("Only n_mels ∈ {80, 128} are supported, got $n_mels")
end

# ── STFT ───────────────────────────────────────────────────────────────────

"""
    stft(audio::Vector{Float32}; n_fft=N_FFT, hop_length=HOP_LENGTH) -> Matrix{Float32}

Compute the power spectrogram (magnitude squared) of `audio`.
Returns a (n_fft÷2+1) × n_frames real matrix.

Matches `torch.stft(..., return_complex=True)[..., :-1].abs() ** 2` in Whisper.
"""
function stft_power(audio::Vector{Float32};
                    n_fft::Int=N_FFT,
                    hop_length::Int=HOP_LENGTH)

    # Hann window – matches PyTorch's `torch.hann_window(N_FFT)`
    window = Float32[0.5f0 * (1.0f0 - cos(2π * n / n_fft)) for n in 0:(n_fft - 1)]

    n_freqs = n_fft ÷ 2 + 1
    # Whisper drops the last frame: same as `stft[..., :-1]`
    # We centre-pad the audio by n_fft÷2 on each side (reflect), then
    # compute exactly the frames that Whisper does.
    pad = n_fft ÷ 2
    padded = vcat(reverse(audio[1:pad]), audio, reverse(audio[end-pad+1:end]))

    n_frames = (length(padded) - n_fft) ÷ hop_length  # drop last frame
    power    = Matrix{Float32}(undef, n_freqs, n_frames)

    buf = Vector{ComplexF32}(undef, n_fft)
    for t in 1:n_frames
        start = (t - 1) * hop_length + 1
        frame = padded[start : start + n_fft - 1] .* window
        frame_c = Complex{Float32}.(frame)
        fft_out = FFTW.fft(frame_c)
        for k in 1:n_freqs
            power[k, t] = abs2(fft_out[k])
        end
    end

    return power
end

# ── Log-Mel Spectrogram ────────────────────────────────────────────────────

"""
    log_mel_spectrogram(audio::Vector{Float32}; n_mels=80) -> Matrix{Float32}

Compute the log-Mel spectrogram for `audio` (16 kHz, Float32 mono).
Returns a (n_mels × N_FRAMES) matrix normalised to the range [-1, 1]
as expected by the Whisper encoder.

Replicates the last three lines of `whisper/audio.py::log_mel_spectrogram`.
"""
function log_mel_spectrogram(audio::Vector{Float32}; n_mels::Int=80)
    power   = stft_power(audio)                         # (n_freqs, n_frames)
    filters = get_mel_filters(n_mels)                   # (n_mels, n_freqs)
    mel_spec = filters * power                           # (n_mels, n_frames)

    log_spec = log10.(max.(mel_spec, 1f-10))

    # Whisper-specific normalisation
    log_spec = max.(log_spec, maximum(log_spec) - 8.0f0)
    log_spec = (log_spec .+ 4.0f0) ./ 4.0f0

    return log_spec
end

"""
    pad_or_trim(audio::Vector{Float32}) -> Vector{Float32}

Pad or trim the waveform to exactly N_SAMPLES (30 seconds at 16 kHz).
"""
function pad_or_trim(audio::Vector{Float32})
    n = length(audio)
    if n > N_SAMPLES
        return audio[1:N_SAMPLES]
    elseif n < N_SAMPLES
        return vcat(audio, zeros(Float32, N_SAMPLES - n))
    end
    return audio
end

"""
    load_audio_file(path::String) -> Vector{Float32}

Load an audio file as a 16 kHz mono Float32 waveform using ffmpeg.
Returns a Vector{Float32} with values in [-1, 1].
"""
function load_audio_file(path::String)
    cmd = `ffmpeg -nostdin -threads 0 -i $path -f f32le -ac 1 -ar $(SAMPLE_RATE) -`
    out = read(cmd)
    return reinterpret(Float32, out)
end

# ─────────────────────────────────────────────────────────────────────────────

# Encoder Constants
const D_MODEL = 384
const ENC_LAYERS = 4
const ENC_FFN_DIM = 1536
const HEADS = 6
const HEAD_DIM = D_MODEL ÷ HEADS
const N_MEL_BINS = 80

# Decoder Constants
const VOCAB_SIZE = 51864
const DEC_LAYERS = 4
const DEC_FFN_DIM = 1536
const MAX_TARGET_POSITION = 448

function sinusoids(channels::Int, length::Int, cx::Luminal.Graph)
    max_timescale = 10000.0f0
    log_timescale_increment = log(max_timescale) / (channels / 2 - 1)
    
    # inv_timescales: e^(-log_inc * i)
    inv_ts = Float32[exp(-log_timescale_increment * i) for i in 0:(channels÷2)-1]
    inv_timescales = Luminal.tensor(cx, inv_ts) # [channels/2]
    
    # scaled_time = arange(length) * inv_timescales
    # arange: [length] -> expand to [length, channels/2]
    # inv_ts: [channels/2] -> expand to [length, channels/2]
    
    # For now arange needs a concrete size or symbol support:
    time_seq = Luminal.arange(cx, length)
    
    # Broadcasting: (length, 1) * (1, channels/2) = (length, channels/2)
    time_exp = Luminal.expand(time_seq, 2, channels ÷ 2)
    inv_exp = Luminal.expand(inv_timescales, 1, length)
    
    scaled_time = time_exp * inv_exp
    
    # concat sin and cos
    return Luminal.concat_along(Luminal.sin(scaled_time), Luminal.cos(scaled_time), 2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Whisper Architecture definitions
# Whisper Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

struct WhisperSelfAttention
    q_proj::Linear
    k_proj::Linear
    v_proj::Linear
    o_proj::Linear
end

function WhisperSelfAttention(hidden::Int, cx::Luminal.Graph,
                               reg=nothing, prefix::String="attn")
    return WhisperSelfAttention(
        _linear(hidden, hidden, cx, reg, "$(prefix).q_proj"; bias=true),
        _linear(hidden, hidden, cx, reg, "$(prefix).k_proj"; bias=false),
        _linear(hidden, hidden, cx, reg, "$(prefix).v_proj"; bias=true),
        _linear(hidden, hidden, cx, reg, "$(prefix).out_proj"; bias=true),
    )
end

function (sa::WhisperSelfAttention)(x::Luminal.GraphTensor, mask::Bool, cache=nothing)
    batch, seq, hidden = Luminal.realized_dims(x.shape)
    scale = (Float32(hidden) / Float32(HEADS)) ^ -0.25f0

    queries = Luminal.permute(Luminal.reshape(sa.q_proj(x) * scale, [batch, seq, HEADS, HEAD_DIM]), [1, 3, 2, 4])
    keys    = Luminal.contiguous(Luminal.permute(Luminal.reshape(sa.k_proj(x) * scale, [batch, seq, HEADS, HEAD_DIM]), [1, 3, 4, 2]))
    values  = Luminal.permute(Luminal.reshape(sa.v_proj(x), [batch, seq, HEADS, HEAD_DIM]), [1, 3, 2, 4])

    weights = Luminal.matmul(queries, keys)

    if mask
        attention_mask = Luminal.triu(x.graph_ref, seq, 1) * -1f9
        mask_expanded = Luminal.expand(Luminal.expand(attention_mask, 1, HEADS), 1, batch)
        weights = weights + mask_expanded
    end

    probs = Luminal.softmax(weights, 4)
    out   = Luminal.matmul(probs, values)
    out   = Luminal.reshape(Luminal.permute(out, [1, 3, 2, 4]), [batch, seq, hidden])

    return sa.o_proj(out), (keys, values)
end

# ─────────────────────────────────────────────────────────────────────────────
# Whisper Cross-Attention
# ─────────────────────────────────────────────────────────────────────────────

struct WhisperCrossAttention
    q_proj::Linear
    k_proj::Linear
    v_proj::Linear
    o_proj::Linear
end

function WhisperCrossAttention(hidden::Int, cx::Luminal.Graph,
                                reg=nothing, prefix::String="cross_attn")
    return WhisperCrossAttention(
        _linear(hidden, hidden, cx, reg, "$(prefix).q_proj"; bias=true),
        _linear(hidden, hidden, cx, reg, "$(prefix).k_proj"; bias=false),
        _linear(hidden, hidden, cx, reg, "$(prefix).v_proj"; bias=true),
        _linear(hidden, hidden, cx, reg, "$(prefix).out_proj"; bias=true),
    )
end

function (ca::WhisperCrossAttention)(queries_t::Luminal.GraphTensor,
                                     keys_t::Luminal.GraphTensor,
                                     values_t::Luminal.GraphTensor)
    batch, dec_seq, hidden = Luminal.realized_dims(queries_t.shape)
    _, enc_seq, _          = Luminal.realized_dims(keys_t.shape)

    scale   = (Float32(hidden) / Float32(HEADS)) ^ -0.25f0
    queries = Luminal.permute(Luminal.reshape(ca.q_proj(queries_t) * scale, [batch, dec_seq, HEADS, HEAD_DIM]), [1, 3, 2, 4])
    keys    = Luminal.contiguous(Luminal.permute(Luminal.reshape(ca.k_proj(keys_t) * scale, [batch, enc_seq, HEADS, HEAD_DIM]), [1, 3, 4, 2]))
    values  = Luminal.permute(Luminal.reshape(ca.v_proj(values_t), [batch, enc_seq, HEADS, HEAD_DIM]), [1, 3, 2, 4])

    weights = Luminal.matmul(queries, keys)
    probs   = Luminal.softmax(weights, 4)
    out     = Luminal.matmul(probs, values)
    out     = Luminal.reshape(Luminal.permute(out, [1, 3, 2, 4]), [batch, dec_seq, hidden])

    return ca.o_proj(out), (keys, values)
end

# ─────────────────────────────────────────────────────────────────────────────
# Encoder Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

struct EncoderTransformerBlock
    attn::WhisperSelfAttention
    attn_norm::LayerNorm
    ff1::Linear
    ff2::Linear
    ff_norm::LayerNorm
end

function EncoderTransformerBlock(hidden::Int, ff::Int, cx::Luminal.Graph,
                                  reg=nothing, prefix::String="block")
    return EncoderTransformerBlock(
        WhisperSelfAttention(hidden, cx, reg, "$(prefix).attn"),
        _layernorm(hidden, cx, reg, "$(prefix).attn_layer_norm"),
        _linear(hidden, ff,     cx, reg, "$(prefix).mlp.fc1"),
        _linear(ff,     hidden, cx, reg, "$(prefix).mlp.fc2"),
        _layernorm(hidden, cx, reg, "$(prefix).final_layer_norm"),
    )
end

function (etb::EncoderTransformerBlock)(x::Luminal.GraphTensor)
    normed  = etb.attn_norm(x)
    y, _    = etb.attn(normed, false)
    x       = x + y
    normed_ff = etb.ff_norm(x)
    y = etb.ff2(Luminal.gelu(etb.ff1(normed_ff)))
    return x + y
end

# ─────────────────────────────────────────────────────────────────────────────
# Audio Encoder
# ─────────────────────────────────────────────────────────────────────────────

struct AudioEncoder
    conv1::Conv1D
    conv2::Conv1D
    layers::Vector{EncoderTransformerBlock}
    post_ln::LayerNorm
end

"""
    AudioEncoder(cx; reg=nothing)

Build the Whisper AudioEncoder.  Pass a `WeightRegistry` to register all
parameter tensors with their HuggingFace safetensors key names.
"""
function AudioEncoder(cx::Luminal.Graph; reg=nothing)
    pfx = "model.encoder"
    layers = [
        EncoderTransformerBlock(D_MODEL, ENC_FFN_DIM, cx, reg,
                                "$(pfx).layers.$(i-1)")
        for i in 1:ENC_LAYERS
    ]
    enc = AudioEncoder(
        _conv1d(N_MEL_BINS, D_MODEL, 3, cx, reg, "$(pfx).conv1"; stride=1, padding=1),
        _conv1d(D_MODEL,    D_MODEL, 3, cx, reg, "$(pfx).conv2"; stride=2, padding=1),
        layers,
        _layernorm(D_MODEL, cx, reg, "$(pfx).layer_norm"),
    )
    return enc
end

function (ae::AudioEncoder)(x::Luminal.GraphTensor)
    _, _, seq = Luminal.realized_dims(x.shape)

    x = ae.conv1(x)
    x = Luminal.gelu(x)
    x = ae.conv2(x)
    x = Luminal.gelu(x)
    x = Luminal.permute(x, [1, 3, 2])

    _, seq_out, _ = Luminal.realized_dims(x.shape)
    seq_div_2 = typeof(seq_out) == Int ? seq_out : 1

    pos_embs = sinusoids(D_MODEL, seq_div_2, x.graph_ref)
    batch    = Luminal.realized_dims(x.shape)[1]
    pos_embs = Luminal.expand(pos_embs, 1, batch)

    x = x + pos_embs
    for layer in ae.layers
        x = layer(x)
    end
    return ae.post_ln(x)
end

# ─────────────────────────────────────────────────────────────────────────────
# Decoder Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

struct DecoderTransformerBlock
    attn::WhisperSelfAttention
    attn_norm::LayerNorm
    cross_attn::WhisperCrossAttention
    cross_attn_norm::LayerNorm
    ff1::Linear
    ff2::Linear
    ff_norm::LayerNorm
end

function DecoderTransformerBlock(hidden::Int, ff::Int, cx::Luminal.Graph,
                                  reg=nothing, prefix::String="block")
    return DecoderTransformerBlock(
        WhisperSelfAttention(hidden, cx, reg, "$(prefix).self_attn"),
        _layernorm(hidden, cx, reg, "$(prefix).self_attn_layer_norm"),
        WhisperCrossAttention(hidden, cx, reg, "$(prefix).encoder_attn"),
        _layernorm(hidden, cx, reg, "$(prefix).encoder_attn_layer_norm"),
        _linear(hidden, ff,     cx, reg, "$(prefix).fc1"),
        _linear(ff,     hidden, cx, reg, "$(prefix).fc2"),
        _layernorm(hidden, cx, reg, "$(prefix).final_layer_norm"),
    )
end

function (dtb::DecoderTransformerBlock)(x::Luminal.GraphTensor, encoded::Luminal.GraphTensor)
    normed    = dtb.attn_norm(x)
    y, _      = dtb.attn(normed, true)
    x         = x + y
    normed_cr = dtb.cross_attn_norm(x)
    y, _      = dtb.cross_attn(normed_cr, encoded, encoded)
    x         = x + y
    normed_ff = dtb.ff_norm(x)
    y = dtb.ff2(Luminal.gelu(dtb.ff1(normed_ff)))
    return x + y
end

# ─────────────────────────────────────────────────────────────────────────────
# Text Decoder
# ─────────────────────────────────────────────────────────────────────────────

struct TextDecoder
    embedding::Embedding
    pos_embedding::Luminal.GraphTensor
    layers::Vector{DecoderTransformerBlock}
    layer_norm::LayerNorm
end

"""
    TextDecoder(cx; reg=nothing)

Build the Whisper TextDecoder.  Pass a `WeightRegistry` to register all
parameter tensors with their HuggingFace safetensors key names.
"""
function TextDecoder(cx::Luminal.Graph; reg=nothing)
    pfx = "model.decoder"
    emb_weight = Luminal.tensor(cx, [VOCAB_SIZE, D_MODEL])
    pos_emb    = Luminal.tensor(cx, [MAX_TARGET_POSITION, D_MODEL])

    if reg !== nothing
        register_weight!(reg, "$(pfx).embed_tokens.weight", emb_weight)
        register_weight!(reg, "$(pfx).embed_positions.weight", pos_emb)
    end

    layers = [
        DecoderTransformerBlock(D_MODEL, DEC_FFN_DIM, cx, reg,
                                "$(pfx).layers.$(i-1)")
        for i in 1:DEC_LAYERS
    ]

    return TextDecoder(
        Embedding(emb_weight),          # reuse registered tensor
        pos_emb,
        layers,
        _layernorm(D_MODEL, cx, reg, "$(pfx).layer_norm"),
    )
end

function (td::TextDecoder)(enc_output::Luminal.GraphTensor, input::Luminal.GraphTensor)
    x = td.embedding(input)

    _, cur_seq = Luminal.realized_dims(input.shape)
    cur_seq_int = typeof(cur_seq) == Int ? cur_seq : 1

    pos_emb_sliced = Luminal.contiguous(Luminal.slice_along(td.pos_embedding, 1, 0, cur_seq_int))
    x = x + Luminal.expand(pos_emb_sliced, 1, Luminal.realized_dims(x.shape)[1])

    for layer in td.layers
        x = layer(x, enc_output)
    end

    out    = td.layer_norm(x)
    logits = Luminal.matmul(out, Luminal.permute(td.embedding.weight, [2, 1]))
    return logits
end

# ─────────────────────────────────────────────────────────────────────────────
# KV Cache Infrastructure
#
# Design: "Incremental Decode Graph"
# The past KV cache is a pre-allocated input tensor (batch, heads, max_seq, head_dim).
# Each decode step, new K/V is written at position `step_pos` using
# slice_along + concat_along (the graph's existing pad/slice ops).
# The updated cache is an output the host feeds back at the next step.
# ─────────────────────────────────────────────────────────────────────────────

"""
    KVCacheState

Host-side state manager for one decode session.
"""
mutable struct KVCacheState
    step_pos::Int
    max_seq::Int
    self_cache::Vector{Tuple{Array{Float32,4}, Array{Float32,4}}}   # per layer
    cross_cache::Vector{Tuple{Array{Float32,4}, Array{Float32,4}}}  # per layer (fixed)
end

"""
    KVCacheState(batch, n_layers, n_heads, head_dim, enc_seq; max_seq=MAX_TARGET_POSITION)

Allocate a zeroed KV cache for a new decode session.
"""
function KVCacheState(batch::Int, n_layers::Int, n_heads::Int, head_dim::Int, enc_seq::Int;
                      max_seq::Int=MAX_TARGET_POSITION)
    self  = [(zeros(Float32, batch, n_heads, max_seq, head_dim),
              zeros(Float32, batch, n_heads, max_seq, head_dim))
             for _ in 1:n_layers]
    cross = [(zeros(Float32, batch, n_heads, enc_seq, head_dim),
              zeros(Float32, batch, n_heads, enc_seq, head_dim))
             for _ in 1:n_layers]
    return KVCacheState(0, max_seq, self, cross)
end

# ─────────────────────────────────────────────────────────────────────────────
# Graph-level cached attention helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _kv_scatter(past, new_slot, step_pos)

Insert `new_slot` (shape …×1×D) into `past` (…×max_seq×D) at index `step_pos`
using slice-and-concat operations that are native to the Luminal graph.
"""
function _kv_scatter(past::Luminal.GraphTensor, new_slot::Luminal.GraphTensor, step_pos::Int)
    max_seq = Luminal.realized_dims(past.shape)[3]
    suffix_len = max_seq - step_pos - 1

    if step_pos == 0
        return Luminal.concat_along(new_slot,
                   Luminal.slice_along(past, 3, 1, max_seq), 3)
    elseif suffix_len == 0
        return Luminal.concat_along(
                   Luminal.slice_along(past, 3, 0, step_pos),
                   new_slot, 3)
    else
        return Luminal.concat_along(
                   Luminal.concat_along(
                       Luminal.slice_along(past, 3, 0, step_pos),
                       new_slot, 3),
                   Luminal.slice_along(past, 3, step_pos + 1, max_seq), 3)
    end
end

"""
    whisper_self_attn_cached(sa, x, step_pos, past_k, past_v)

Single-token cached self-attention.

- `x`       : (batch, 1, hidden)
- `step_pos`: current 0-indexed decode position
- `past_k`  : (batch, heads, max_seq, head_dim)
- `past_v`  : (batch, heads, max_seq, head_dim)

Returns `(output, new_k, new_v)`.
"""
function whisper_self_attn_cached(sa::WhisperSelfAttention,
                                   x::Luminal.GraphTensor,
                                   step_pos::Int,
                                   past_k::Luminal.GraphTensor,
                                   past_v::Luminal.GraphTensor)
    batch, _, hidden = Luminal.realized_dims(x.shape)
    scale = (Float32(hidden) / Float32(HEADS)) ^ -0.25f0

    # Project: (batch, 1, hidden) → (batch, heads, 1, head_dim)
    q = Luminal.permute(
        Luminal.reshape(sa.q_proj(x) * scale, [batch, 1, HEADS, HEAD_DIM]),
        [1, 3, 2, 4])
    # k_new: (batch, heads, 1, head_dim) after fixing permute order
    k_new = Luminal.contiguous(
        Luminal.permute(
            Luminal.reshape(sa.k_proj(x) * scale, [batch, 1, HEADS, HEAD_DIM]),
            [1, 3, 2, 4]))
    v_new = Luminal.permute(
        Luminal.reshape(sa.v_proj(x), [batch, 1, HEADS, HEAD_DIM]),
        [1, 3, 2, 4])

    # Update cache
    new_k = _kv_scatter(past_k, k_new, step_pos)
    new_v = _kv_scatter(past_v, v_new, step_pos)

    # Attend over context [0 : step_pos+1]
    k_ctx   = Luminal.slice_along(new_k, 3, 0, step_pos + 1)   # (b, h, ctx, d)
    v_ctx   = Luminal.slice_along(new_v, 3, 0, step_pos + 1)   # (b, h, ctx, d)
    k_ctx_t = Luminal.contiguous(Luminal.permute(k_ctx, [1, 2, 4, 3]))  # (b, h, d, ctx)

    weights = Luminal.matmul(q, k_ctx_t)     # (b, h, 1, ctx)
    probs   = Luminal.softmax(weights, 4)
    out     = Luminal.matmul(probs, v_ctx)   # (b, h, 1, d)
    out     = Luminal.reshape(Luminal.permute(out, [1, 3, 2, 4]), [batch, 1, hidden])

    return sa.o_proj(out), new_k, new_v
end

"""
    whisper_cross_attn_cached(ca, x, enc_k, enc_v)

Single-token cross-attention using the pre-computed encoder K/V.
- `enc_k`, `enc_v`: (batch, heads, enc_seq, head_dim)
"""
function whisper_cross_attn_cached(ca::WhisperCrossAttention,
                                    x::Luminal.GraphTensor,
                                    enc_k::Luminal.GraphTensor,
                                    enc_v::Luminal.GraphTensor)
    batch, _, hidden = Luminal.realized_dims(x.shape)
    scale = (Float32(hidden) / Float32(HEADS)) ^ -0.25f0

    q   = Luminal.permute(
              Luminal.reshape(ca.q_proj(x) * scale, [batch, 1, HEADS, HEAD_DIM]),
              [1, 3, 2, 4])                              # (b, h, 1, d)
    k_t = Luminal.contiguous(Luminal.permute(enc_k, [1, 2, 4, 3]))  # (b, h, d, enc_seq)

    weights = Luminal.matmul(q, k_t)         # (b, h, 1, enc_seq)
    probs   = Luminal.softmax(weights, 4)
    out     = Luminal.matmul(probs, enc_v)   # (b, h, 1, d)
    out     = Luminal.reshape(Luminal.permute(out, [1, 3, 2, 4]), [batch, 1, hidden])
    return ca.o_proj(out)
end

# ─────────────────────────────────────────────────────────────────────────────
# Incremental Decode Step Graph
# ─────────────────────────────────────────────────────────────────────────────

"""
    IncrementalDecodeGraph

Node IDs for driving one step of the incremental decode graph.
"""
struct IncrementalDecodeGraph
    token_input_id::Int
    cross_k_ids::Vector{Int}          # encoder K inputs, per layer
    cross_v_ids::Vector{Int}          # encoder V inputs, per layer
    self_k_ids::Vector{Int}           # past self-K inputs, per layer
    self_v_ids::Vector{Int}           # past self-V inputs, per layer
    logits_id::Int                    # output logits
    new_self_k_ids::Vector{Int}       # output updated self-K, per layer
    new_self_v_ids::Vector{Int}       # output updated self-V, per layer
    step_pos::Base.RefValue{Int}      # current decode position (host-managed)
end

"""
    build_decode_step!(td, graph, enc_seq, step_pos; max_seq, batch)

Build the single-step incremental decode graph inside `graph`.

One graph must be built PER step_pos value (since `step_pos` is baked into
the slice positions). For typical use, build up to `max_decode_steps` graphs
ahead of time, or rebuild lazily per step.

Returns an `IncrementalDecodeGraph`.
"""
function build_decode_step!(td::TextDecoder, graph::Luminal.Graph,
                             enc_seq::Int, step_pos::Int;
                             max_seq::Int=MAX_TARGET_POSITION,
                             batch::Int=1)
    n_layers = length(td.layers)

    # ── Define inputs ────────────────────────────────────────────────────────
    token_in = Luminal.tensor(graph, [batch, 1])

    cross_k_tensors = [Luminal.tensor(graph, [batch, HEADS, enc_seq, HEAD_DIM])
                       for _ in 1:n_layers]
    cross_v_tensors = [Luminal.tensor(graph, [batch, HEADS, enc_seq, HEAD_DIM])
                       for _ in 1:n_layers]
    self_k_tensors  = [Luminal.tensor(graph, [batch, HEADS, max_seq, HEAD_DIM])
                       for _ in 1:n_layers]
    self_v_tensors  = [Luminal.tensor(graph, [batch, HEADS, max_seq, HEAD_DIM])
                       for _ in 1:n_layers]

    # ── Embedding + positional ───────────────────────────────────────────────
    x = td.embedding(token_in)                                  # (batch, 1, D_MODEL)
    pos_slot = Luminal.contiguous(
        Luminal.slice_along(td.pos_embedding, 1, step_pos, step_pos + 1))
    x = x + Luminal.expand(pos_slot, 1, batch)                 # add positional

    # ── Decoder layers ───────────────────────────────────────────────────────
    new_self_k_tensors = Luminal.GraphTensor[]
    new_self_v_tensors = Luminal.GraphTensor[]

    for (i, layer) in enumerate(td.layers)
        normed        = layer.attn_norm(x)
        attn_out, nk, nv = whisper_self_attn_cached(
            layer.attn, normed, step_pos,
            self_k_tensors[i], self_v_tensors[i])
        x = x + attn_out
        push!(new_self_k_tensors, nk)
        push!(new_self_v_tensors, nv)

        normed_cr = layer.cross_attn_norm(x)
        cross_out = whisper_cross_attn_cached(
            layer.cross_attn, normed_cr,
            cross_k_tensors[i], cross_v_tensors[i])
        x = x + cross_out

        normed_ff = layer.ff_norm(x)
        y = layer.ff2(Luminal.gelu(layer.ff1(normed_ff)))
        x = x + y
    end

    out    = td.layer_norm(x)                                   # (batch, 1, D_MODEL)
    logits = Luminal.matmul(
        Luminal.contiguous(Luminal.slice_along(out, 2, 0, 1)),
        Luminal.permute(td.embedding.weight, [2, 1]))           # (batch, 1, vocab)

    return IncrementalDecodeGraph(
        token_in.id,
        [t.id for t in cross_k_tensors],
        [t.id for t in cross_v_tensors],
        [t.id for t in self_k_tensors],
        [t.id for t in self_v_tensors],
        logits.id,
        [t.id for t in new_self_k_tensors],
        [t.id for t in new_self_v_tensors],
        Ref(step_pos))
end

# ─────────────────────────────────────────────────────────────────────────────
# Host-side decode step helper
# ─────────────────────────────────────────────────────────────────────────────

"""
    decode_step!(exec_fn, idg, cache, token_ids, cross_kv_arrays; device=nothing)

Execute one step of the incremental decode graph.

- `exec_fn`        : compiled execution function
- `idg`            : `IncrementalDecodeGraph` built for this `step_pos`
- `cache`          : `KVCacheState`  (mutated: self_cache updated, step_pos incremented)
- `token_ids`      : `Matrix{Float32}` of shape (batch, 1)
- `cross_kv_arrays`: `Vector` of `(k_array, v_array)` per layer (from encoder precompute)

Returns `logits::Array{Float32}` of shape (batch, 1, vocab_size).
"""
function decode_step!(exec_obj::Any,
                      idg::IncrementalDecodeGraph,
                      cache::KVCacheState,
                      token_ids::Matrix{Float32},
                      cross_kv_arrays::Vector;
                      device=Luminal.get_device())
    inputs = Dict{Int, Any}()
    inputs[idg.token_input_id] = token_ids

    for (i, (ck, cv)) in enumerate(cross_kv_arrays)
        inputs[idg.cross_k_ids[i]] = ck
        inputs[idg.cross_v_ids[i]] = cv
    end

    for i in eachindex(idg.self_k_ids)
        inputs[idg.self_k_ids[i]] = cache.self_cache[i][1]
        inputs[idg.self_v_ids[i]] = cache.self_cache[i][2]
    end

    all_output_ids = vcat(
        [idg.logits_id],
        idg.new_self_k_ids,
        idg.new_self_v_ids)

    result = if exec_obj isa Luminal.Graph
        Luminal.execute(exec_obj, all_output_ids, inputs, device)
    else
        # CompiledGraph returns results vector indexed by node_id
        res_vec = exec_obj(inputs, device)
        Dict{Int, Any}(id => res_vec[id] for id in all_output_ids)
    end

    for i in eachindex(idg.new_self_k_ids)
        cache.self_cache[i] = (
            result[idg.new_self_k_ids[i]],
            result[idg.new_self_v_ids[i]])
    end

    cache.step_pos += 1
    return result[idg.logits_id]
end

"""
    project_cross_kv(td::TextDecoder, graph::Luminal.Graph, enc_output::Luminal.GraphTensor)

Compute the cross-attention K and V projections for all decoder layers.
Returns `Vector{Tuple{GraphTensor, GraphTensor}}`.
"""
function project_cross_kv(td::TextDecoder, graph::Luminal.Graph, enc_output::Luminal.GraphTensor)
    batch, seq, d_model = Luminal.realized_dims(enc_output.shape)
    kv = Tuple{Luminal.GraphTensor, Luminal.GraphTensor}[]
    
    scale = (Float32(d_model) / Float32(HEADS)) ^ -0.25f0
    
    for layer in td.layers
        ca = layer.cross_attn
        
        # Project K: (batch, seq, hidden) -> (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        k = Luminal.permute(
            Luminal.reshape(ca.k_proj(enc_output) * scale, [batch, seq, HEADS, HEAD_DIM]),
            [1, 3, 2, 4])
        
        # Project V: (batch, seq, hidden) -> (batch, heads, seq, head_dim)
        v = Luminal.permute(
            Luminal.reshape(ca.v_proj(enc_output), [batch, seq, HEADS, HEAD_DIM]),
            [1, 3, 2, 4])
            
        push!(kv, (Luminal.contiguous(k), Luminal.contiguous(v)))
    end
    return kv
end

export KVCacheState, IncrementalDecodeGraph, build_decode_step!, decode_step!,
       whisper_self_attn_cached, whisper_cross_attn_cached, project_cross_kv,
       D_MODEL, HEADS, HEAD_DIM, MAX_TARGET_POSITION, VOCAB_SIZE, DEC_LAYERS


