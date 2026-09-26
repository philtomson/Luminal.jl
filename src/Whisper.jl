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

# Slaney mel scale (librosa's default, used by Whisper): linear below 1 kHz,
# logarithmic above.
const _MEL_F_SP     = 200.0 / 3
const _MEL_MIN_LOG  = 1000.0
const _MEL_LOGSTEP  = log(6.4) / 27.0
const _MEL_MIN_LOGM = _MEL_MIN_LOG / _MEL_F_SP

"""
    hz_to_mel(hz)

Convert a frequency in Hz to the Slaney Mel scale used by Whisper.
"""
hz_to_mel(hz::Real) = hz < _MEL_MIN_LOG ? hz / _MEL_F_SP :
                      _MEL_MIN_LOGM + log(hz / _MEL_MIN_LOG) / _MEL_LOGSTEP

"""
    mel_to_hz(mel)

Convert a Slaney Mel-scale value back to Hz.
"""
mel_to_hz(mel::Real) = mel < _MEL_MIN_LOGM ? mel * _MEL_F_SP :
                       _MEL_MIN_LOG * exp(_MEL_LOGSTEP * (mel - _MEL_MIN_LOGM))

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

    # Slaney normalization (librosa norm="slaney"): scale each filter by
    # 2 / its width in Hz, so every filter has unit area
    for m in 1:n_mels
        width_hz = mel_points[m + 2] - mel_points[m]
        if width_hz > 0
            filters[m, :] .*= Float32(2.0 / width_hz)
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
    # Reflect padding without repeating the edge sample (torch/numpy "reflect")
    padded = vcat(audio[pad+1:-1:2], audio, audio[end-1:-1:end-pad])

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
    cmd = `ffmpeg -nostdin -loglevel error -threads 0 -i $path -f f32le -ac 1 -ar $(SAMPLE_RATE) -`
    out = read(cmd)
    return collect(reinterpret(Float32, out))
end

# ─────────────────────────────────────────────────────────────────────────────

# Model constants (openai/whisper-tiny). Other sizes are built by passing the
# dimensions to `AudioEncoder` / `TextDecoder`.
const D_MODEL = 384
const ENC_LAYERS = 4
const ENC_FFN_DIM = 1536
const HEADS = 6
const HEAD_DIM = D_MODEL ÷ HEADS
const N_MEL_BINS = 80
const MAX_SOURCE_POSITION = 1500

const VOCAB_SIZE = 51865
const DEC_LAYERS = 4
const DEC_FFN_DIM = 1536
const MAX_TARGET_POSITION = 448

# Activations use the (Hidden, Seq, Batch) layout throughout; attention heads are
# (HeadDim, Seq, Heads, Batch). The encoder input is the log-mel spectrogram as
# (n_mels, frames, batch).

# Whisper's activation is the exact (erf) GELU.
_gelu(x::Luminal.GraphTensor) = Luminal.gelu(x; approximate=false)

# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────

"""
    WhisperAttention(hidden, n_heads, cx, reg=nothing, prefix="self_attn")

Multi-head attention with Whisper's projections (`k_proj` has no bias). Used for
the encoder and decoder self-attention and for the decoder's cross-attention.
"""
struct WhisperAttention
    q_proj::Linear
    k_proj::Linear
    v_proj::Linear
    out_proj::Linear
    n_heads::Int
end

function WhisperAttention(hidden::Int, n_heads::Int, cx::Luminal.Graph,
                          reg=nothing, prefix::String="self_attn")
    return WhisperAttention(
        _linear(hidden, hidden, cx, reg, "$(prefix).q_proj"; bias=true),
        _linear(hidden, hidden, cx, reg, "$(prefix).k_proj"; bias=false),
        _linear(hidden, hidden, cx, reg, "$(prefix).v_proj"; bias=true),
        _linear(hidden, hidden, cx, reg, "$(prefix).out_proj"; bias=true),
        n_heads)
end

const WhisperSelfAttention  = WhisperAttention
const WhisperCrossAttention = WhisperAttention

# (Hidden, S, B) -> (HeadDim, S, Heads, B). Projections are packed head-major, so
# the head dimension is the fastest-varying one.
function _split_heads(x::Luminal.GraphTensor, n_heads::Int)
    hidden, s, b = Luminal.realized_dims(x.shape)
    return Luminal.permute(Luminal.reshape(x, [hidden ÷ n_heads, n_heads, s, b]), [1, 3, 2, 4])
end

# Scaled dot-product attention over (D, S, H, B) heads -> (D*H, Sq, B). The scores
# are laid out (Sk, Sq, H, B), so the softmax runs along the contiguous first dim as
# one fused kernel, and V * probs comes out directly as (D, Sq, H, B).
function _attend(q::Luminal.GraphTensor, k::Luminal.GraphTensor, v::Luminal.GraphTensor;
                 causal::Bool=false)
    d, sq, h, b = Luminal.realized_dims(q.shape)
    scale = 1.0f0 / sqrt(Float32(d))
    scores = Luminal.matmul(Luminal.permute(k, [2, 1, 3, 4]), q) * scale      # (Sk, Sq, H, B)
    if causal && sq > 1
        # key j > query i is masked: triu marks col > row, so transpose it
        mask = Luminal.permute(Luminal.triu(q.graph_ref, sq, 1), [2, 1]) * -1f9
        scores = scores + Luminal.expand(Luminal.expand(mask, 3, h), 4, b)
    end
    probs = Luminal.softmax1(scores)
    out = Luminal.matmul(v, probs)                                            # (D, Sq, H, B)
    return Luminal.reshape(Luminal.permute(out, [1, 3, 2, 4]), [d * h, sq, b])
end

"""
    (a::WhisperAttention)(x, kv=x; causal=false)

`x`: (Hidden, Sq, B) queries; `kv`: (Hidden, Sk, B) keys/values source (the
encoder output for cross-attention).
"""
function (a::WhisperAttention)(x::Luminal.GraphTensor, kv::Luminal.GraphTensor=x;
                               causal::Bool=false)
    q = _split_heads(a.q_proj(x), a.n_heads)
    k = _split_heads(a.k_proj(kv), a.n_heads)
    v = _split_heads(a.v_proj(kv), a.n_heads)
    return a.out_proj(_attend(q, k, v; causal=causal))
end

# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

struct EncoderTransformerBlock
    self_attn::WhisperAttention
    self_attn_layer_norm::LayerNorm
    fc1::Linear
    fc2::Linear
    final_layer_norm::LayerNorm
end

function EncoderTransformerBlock(hidden::Int, n_heads::Int, ff::Int, cx::Luminal.Graph,
                                 reg=nothing, prefix::String="block")
    return EncoderTransformerBlock(
        WhisperAttention(hidden, n_heads, cx, reg, "$(prefix).self_attn"),
        _layernorm(hidden, cx, reg, "$(prefix).self_attn_layer_norm"),
        _linear(hidden, ff,     cx, reg, "$(prefix).fc1"),
        _linear(ff,     hidden, cx, reg, "$(prefix).fc2"),
        _layernorm(hidden, cx, reg, "$(prefix).final_layer_norm"))
end

function (b::EncoderTransformerBlock)(x::Luminal.GraphTensor)
    x = x + b.self_attn(b.self_attn_layer_norm(x))
    return x + b.fc2(_gelu(b.fc1(b.final_layer_norm(x))))
end

struct AudioEncoder
    conv1::Conv1D
    conv2::Conv1D
    embed_positions::Luminal.GraphTensor    # (max_source_positions, Hidden)
    layers::Vector{EncoderTransformerBlock}
    layer_norm::LayerNorm
end

"""
    AudioEncoder(cx; reg=nothing, d_model, n_layers, n_heads, ffn_dim, n_mels, max_positions)

Build the Whisper audio encoder (defaults: whisper-tiny). Pass a `WeightRegistry`
to register every parameter under its Hugging Face safetensors key.
"""
function AudioEncoder(cx::Luminal.Graph; reg=nothing,
                      d_model::Int=D_MODEL, n_layers::Int=ENC_LAYERS, n_heads::Int=HEADS,
                      ffn_dim::Int=ENC_FFN_DIM, n_mels::Int=N_MEL_BINS,
                      max_positions::Int=MAX_SOURCE_POSITION)
    pfx = "model.encoder"
    pos = Luminal.tensor(cx, [max_positions, d_model])
    reg !== nothing && register_weight!(reg, "$(pfx).embed_positions.weight", pos)
    return AudioEncoder(
        _conv1d(n_mels,  d_model, 3, cx, reg, "$(pfx).conv1"; stride=1, padding=1),
        _conv1d(d_model, d_model, 3, cx, reg, "$(pfx).conv2"; stride=2, padding=1),
        pos,
        [EncoderTransformerBlock(d_model, n_heads, ffn_dim, cx, reg, "$(pfx).layers.$(i-1)")
         for i in 1:n_layers],
        _layernorm(d_model, cx, reg, "$(pfx).layer_norm"))
end

# Rows [0, s) of a (positions, Hidden) table as (Hidden, s, 1).
function _positions(table::Luminal.GraphTensor, s)
    return Luminal.permute(Luminal.slice_along(table, 1, 0, s), [2, 1])
end

"""
    (ae::AudioEncoder)(mel)

`mel`: (n_mels, frames, B) log-mel spectrogram -> (Hidden, frames ÷ 2, B).
"""
function (ae::AudioEncoder)(mel::Luminal.GraphTensor)
    x = _gelu(ae.conv1(mel))
    x = _gelu(ae.conv2(x))
    x = x + _positions(ae.embed_positions, Luminal.realized_dims(x.shape)[2])
    for layer in ae.layers
        x = layer(x)
    end
    return ae.layer_norm(x)
end

# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

struct DecoderTransformerBlock
    self_attn::WhisperAttention
    self_attn_layer_norm::LayerNorm
    encoder_attn::WhisperAttention
    encoder_attn_layer_norm::LayerNorm
    fc1::Linear
    fc2::Linear
    final_layer_norm::LayerNorm
end

function DecoderTransformerBlock(hidden::Int, n_heads::Int, ff::Int, cx::Luminal.Graph,
                                 reg=nothing, prefix::String="block")
    return DecoderTransformerBlock(
        WhisperAttention(hidden, n_heads, cx, reg, "$(prefix).self_attn"),
        _layernorm(hidden, cx, reg, "$(prefix).self_attn_layer_norm"),
        WhisperAttention(hidden, n_heads, cx, reg, "$(prefix).encoder_attn"),
        _layernorm(hidden, cx, reg, "$(prefix).encoder_attn_layer_norm"),
        _linear(hidden, ff,     cx, reg, "$(prefix).fc1"),
        _linear(ff,     hidden, cx, reg, "$(prefix).fc2"),
        _layernorm(hidden, cx, reg, "$(prefix).final_layer_norm"))
end

function (b::DecoderTransformerBlock)(x::Luminal.GraphTensor, encoded::Luminal.GraphTensor)
    x = x + b.self_attn(b.self_attn_layer_norm(x); causal=true)
    x = x + b.encoder_attn(b.encoder_attn_layer_norm(x), encoded)
    return x + b.fc2(_gelu(b.fc1(b.final_layer_norm(x))))
end

struct TextDecoder
    embed_tokens::Embedding                   # (Vocab, Hidden), tied to the output head
    embed_positions::Luminal.GraphTensor      # (max_target_positions, Hidden)
    layers::Vector{DecoderTransformerBlock}
    layer_norm::LayerNorm
end

"""
    TextDecoder(cx; reg=nothing, d_model, n_layers, n_heads, ffn_dim, vocab_size, max_positions)

Build the Whisper text decoder (defaults: whisper-tiny). Pass a `WeightRegistry`
to register every parameter under its Hugging Face safetensors key.
"""
function TextDecoder(cx::Luminal.Graph; reg=nothing,
                     d_model::Int=D_MODEL, n_layers::Int=DEC_LAYERS, n_heads::Int=HEADS,
                     ffn_dim::Int=DEC_FFN_DIM, vocab_size::Int=VOCAB_SIZE,
                     max_positions::Int=MAX_TARGET_POSITION)
    pfx = "model.decoder"
    pos = Luminal.tensor(cx, [max_positions, d_model])
    reg !== nothing && register_weight!(reg, "$(pfx).embed_positions.weight", pos)
    return TextDecoder(
        _embedding(vocab_size, d_model, cx, reg, "$(pfx).embed_tokens"),
        pos,
        [DecoderTransformerBlock(d_model, n_heads, ffn_dim, cx, reg, "$(pfx).layers.$(i-1)")
         for i in 1:n_layers],
        _layernorm(d_model, cx, reg, "$(pfx).layer_norm"))
end

"""
    (td::TextDecoder)(encoded, tokens)

Full-sequence (uncached) decoder. `encoded`: (Hidden, S_enc, B) encoder output;
`tokens`: (S, B) 0-indexed token ids. Returns logits (Vocab, S, B).
"""
function (td::TextDecoder)(encoded::Luminal.GraphTensor, tokens::Luminal.GraphTensor)
    x = td.embed_tokens(tokens)
    x = x + _positions(td.embed_positions, Luminal.realized_dims(x.shape)[2])
    for layer in td.layers
        x = layer(x, encoded)
    end
    return Luminal.matmul(td.embed_tokens.weight, td.layer_norm(x))
end

# Dimensions of a built decoder, for rebuilding it in another graph.
function decoder_config(td::TextDecoder)
    vocab, d_model = Luminal.realized_dims(td.embed_tokens.weight.shape)
    layer = td.layers[1]
    return (d_model=d_model, n_layers=length(td.layers), n_heads=layer.self_attn.n_heads,
            ffn_dim=Luminal.realized_dims(layer.fc1.weight.shape)[1], vocab_size=vocab,
            max_positions=Luminal.realized_dims(td.embed_positions.shape)[1])
end

# ─────────────────────────────────────────────────────────────────────────────
# KV-cached incremental decoding
#
# The decode step is one shape-static graph for every position: the position is
# data (a (1,) tensor), self-attention reads the device-resident cache under
# `DecodeAttention`, which also writes each step's K/V slot into the cache in
# place. Cross-attention K/V are projected once from the encoder output.
# ─────────────────────────────────────────────────────────────────────────────

"""
    KVCacheState

Device-resident K/V for one decode session.
- `self_cache[i]`: `(K, V)` for decoder layer i, each (head_dim, max_seq, heads, batch)
- `cross_cache[i]`: `(K, V)` projected from the encoder output, each
  (head_dim, enc_seq, heads, batch)
- `step_pos`: 0-indexed position of the next token
"""
mutable struct KVCacheState{A<:AbstractArray{Float32,4}}
    step_pos::Int
    max_seq::Int
    self_cache::Vector{Tuple{A, A}}
    cross_cache::Vector{Tuple{A, A}}
end

"""
    KVCacheState(n_layers, n_heads, head_dim, enc_seq; batch=1, max_seq=MAX_TARGET_POSITION,
                 device=CPUDevice())

Allocate a zeroed cache.
"""
function KVCacheState(n_layers::Int, n_heads::Int, head_dim::Int, enc_seq::Int;
                      batch::Int=1, max_seq::Int=MAX_TARGET_POSITION,
                      device::Luminal.AbstractDevice=Luminal.CPUDevice())
    z(s) = Luminal.zero_tensor(device, Float32, head_dim, s, n_heads, batch)
    return KVCacheState(0, max_seq,
                        [(z(max_seq), z(max_seq)) for _ in 1:n_layers],
                        [(z(enc_seq), z(enc_seq)) for _ in 1:n_layers])
end

"""
    whisper_self_attn_cached(sa, x, pos_tensor, past_k, past_v)

Single-token causal self-attention against the cache.
- `x`: (Hidden, 1, B); `pos_tensor`: (1,) current position, as data
- `past_k`, `past_v`: (head_dim, max_seq, heads, B); slots `>= pos` are ignored
Returns `(output, k_new, v_new)`, with this token's (head_dim, 1, heads, B) K/V slot.
"""
function whisper_self_attn_cached(sa::WhisperAttention, x::Luminal.GraphTensor,
                                  pos_tensor::Luminal.GraphTensor,
                                  past_k::Luminal.GraphTensor, past_v::Luminal.GraphTensor)
    hidden, _, batch = Luminal.realized_dims(x.shape)
    d = hidden ÷ sa.n_heads
    q     = _split_heads(sa.q_proj(x), sa.n_heads)
    k_new = _split_heads(sa.k_proj(x), sa.n_heads)
    v_new = _split_heads(sa.v_proj(x), sa.n_heads)
    ins = [(t.id, 0, t.shape) for t in (q, past_k, past_v, k_new, v_new, pos_tensor)]
    # The op also writes this token's K/V into the cache slot (write_cache=true).
    out = Luminal.add_op!(x.graph_ref, Luminal.DecodeAttention(1.0f0 / sqrt(Float32(d)), true), ins,
                          Luminal.ShapeTracker([d, sa.n_heads, batch]))
    return sa.out_proj(Luminal.reshape(out, [hidden, 1, batch])), k_new, v_new
end

"""
    whisper_cross_attn_cached(ca, x, enc_k, enc_v)

Single-token cross-attention against the projected encoder K/V.
- `x`: (Hidden, 1, B); `enc_k`, `enc_v`: (head_dim, enc_seq, heads, B)
"""
function whisper_cross_attn_cached(ca::WhisperAttention, x::Luminal.GraphTensor,
                                   enc_k::Luminal.GraphTensor, enc_v::Luminal.GraphTensor)
    hidden, _, batch = Luminal.realized_dims(x.shape)
    d = hidden ÷ ca.n_heads
    q = _split_heads(ca.q_proj(x), ca.n_heads)                            # (D, 1, H, B)
    # With one query every transpose below moves a size-1 dim, so none copies.
    scores = Luminal.matmul(Luminal.permute(q, [2, 1, 3, 4]), enc_k) *
             (1.0f0 / sqrt(Float32(d)))                                   # (1, S, H, B)
    probs = Luminal.softmax(scores, 2)
    out = Luminal.matmul(enc_v, Luminal.permute(probs, [2, 1, 3, 4]))    # (D, 1, H, B)
    out = Luminal.reshape(Luminal.permute(out, [1, 3, 2, 4]), [hidden, 1, batch])
    return ca.out_proj(out)
end

"""
    IncrementalDecodeGraph

Node IDs for driving the incremental decode graph.
"""
struct IncrementalDecodeGraph
    token_input_id::Int
    pos_input_id::Int
    cross_k_ids::Vector{Int}          # encoder K inputs, per layer
    cross_v_ids::Vector{Int}          # encoder V inputs, per layer
    self_k_ids::Vector{Int}           # self-attention K cache inputs, per layer
    self_v_ids::Vector{Int}           # self-attention V cache inputs, per layer
    logits_id::Int                    # (Vocab, 1, B)
    new_self_k_ids::Vector{Int}       # this step's K slot per layer, (D, 1, H, B)
    new_self_v_ids::Vector{Int}       # this step's V slot per layer, (D, 1, H, B)
end

"""
    build_decode_step!(td, graph, enc_seq; max_seq=MAX_TARGET_POSITION, batch=1)

Build the single-token decode graph for `td` (whose weights live in `graph`).
The graph is independent of the position, so it is compiled (and captured) once.
"""
function build_decode_step!(td::TextDecoder, graph::Luminal.Graph, enc_seq::Int;
                            max_seq::Int=MAX_TARGET_POSITION, batch::Int=1)
    cfg = decoder_config(td)
    h, d = cfg.n_heads, cfg.d_model ÷ cfg.n_heads
    n = cfg.n_layers
    token_in = Luminal.tensor(graph, [1, batch])
    pos_in   = Luminal.tensor(graph, [1])
    cross_k  = [Luminal.tensor(graph, [d, enc_seq, h, batch]) for _ in 1:n]
    cross_v  = [Luminal.tensor(graph, [d, enc_seq, h, batch]) for _ in 1:n]
    self_k   = [Luminal.tensor(graph, [d, max_seq, h, batch]) for _ in 1:n]
    self_v   = [Luminal.tensor(graph, [d, max_seq, h, batch]) for _ in 1:n]

    x = td.embed_tokens(token_in)                                              # (Hidden, 1, B)
    x = x + Embedding(td.embed_positions)(Luminal.reshape(pos_in, [1, 1]))     # (Hidden, 1, 1)

    new_k = Luminal.GraphTensor[]
    new_v = Luminal.GraphTensor[]
    for (i, layer) in enumerate(td.layers)
        y, nk, nv = whisper_self_attn_cached(layer.self_attn, layer.self_attn_layer_norm(x),
                                             pos_in, self_k[i], self_v[i])
        x = x + y
        push!(new_k, nk)
        push!(new_v, nv)
        x = x + whisper_cross_attn_cached(layer.encoder_attn, layer.encoder_attn_layer_norm(x),
                                          cross_k[i], cross_v[i])
        x = x + layer.fc2(_gelu(layer.fc1(layer.final_layer_norm(x))))
    end
    logits = Luminal.matmul(td.embed_tokens.weight, td.layer_norm(x))          # (Vocab, 1, B)

    ids(ts) = [t.id for t in ts]
    return IncrementalDecodeGraph(token_in.id, pos_in.id, ids(cross_k), ids(cross_v),
                                  ids(self_k), ids(self_v), logits.id, ids(new_k), ids(new_v))
end

"""
    decode_step!(exec_fn, idg, cache, token_ids; device=get_device())

Run one decode step for `token_ids` (one 0-indexed token per batch entry) at
position `cache.step_pos`, write its K/V into the cache and advance the position.
Returns the (Vocab, 1, B) logits.
"""
function decode_step!(exec_fn, idg::IncrementalDecodeGraph, cache::KVCacheState,
                      token_ids::AbstractVector{<:Integer};
                      device=Luminal.get_device())
    cache.step_pos < cache.max_seq || error("KV cache full ($(cache.max_seq) positions)")
    inputs = Dict{Int, Any}(idg.token_input_id => Float32.(Base.reshape(token_ids, 1, :)),
                            idg.pos_input_id => Float32[cache.step_pos])
    for i in eachindex(idg.self_k_ids)
        inputs[idg.self_k_ids[i]]  = cache.self_cache[i][1]
        inputs[idg.self_v_ids[i]]  = cache.self_cache[i][2]
        inputs[idg.cross_k_ids[i]] = cache.cross_cache[i][1]
        inputs[idg.cross_v_ids[i]] = cache.cross_cache[i][2]
    end
    results = if exec_fn isa Luminal.Graph
        Luminal.execute(exec_fn, vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids),
                        inputs, device)
    else
        exec_fn(inputs; device=device)
    end
    # The step's DecodeAttention nodes wrote this token's K/V into the cache.
    cache.step_pos += 1
    return results[idg.logits_id]
end

"""
    project_cross_kv(td, enc_output) -> Vector{Tuple{GraphTensor, GraphTensor}}

Cross-attention K and V for every decoder layer from the (Hidden, S_enc, B)
encoder output, each as a contiguous (head_dim, S_enc, heads, B) tensor.
"""
function project_cross_kv(td::TextDecoder, enc_output::Luminal.GraphTensor)
    return [(Luminal.contiguous(_split_heads(l.encoder_attn.k_proj(enc_output), l.encoder_attn.n_heads)),
             Luminal.contiguous(_split_heads(l.encoder_attn.v_proj(enc_output), l.encoder_attn.n_heads)))
            for l in td.layers]
end

export KVCacheState, IncrementalDecodeGraph, build_decode_step!, decode_step!,
       whisper_self_attn_cached, whisper_cross_attn_cached, project_cross_kv, decoder_config,
       D_MODEL, HEADS, HEAD_DIM, MAX_TARGET_POSITION, VOCAB_SIZE, DEC_LAYERS
