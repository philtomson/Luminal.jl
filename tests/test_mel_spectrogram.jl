using Test
using Luminal
using Luminal.NN

@testset "Mel Filterbank Shape and Properties" begin
    fb = NN.mel_filters(80)
    @test size(fb) == (80, 201)           # (n_mels, n_fft÷2+1)
    @test eltype(fb) == Float32
    @test all(fb .>= 0)                   # filters are non-negative
    @test !all(fb .== 0)                  # at least some non-zero
    
    fb128 = NN.mel_filters(128)
    @test size(fb128) == (128, 201)
    
    # Cached versions should be identical
    @test NN.get_mel_filters(80) === NN.MEL_FILTERS_80
    @test NN.get_mel_filters(128) === NN.MEL_FILTERS_128
end

@testset "STFT Power Shape and Properties" begin
    # Generate 30 seconds of silence (deterministic)
    audio = zeros(Float32, NN.N_SAMPLES)
    power = NN.stft_power(audio)
    
    @test size(power, 1) == NN.N_FFT ÷ 2 + 1   # 201 freq bins
    @test all(power .>= 0)                        # power is non-negative
    
    # DC bin of silence should be 0
    @test all(power .== 0)
    
    # Test with a pure sine wave at 440 Hz
    t = Float32.(0:(NN.N_SAMPLES - 1)) ./ Float32(NN.SAMPLE_RATE)
    sine = Float32.(sin.(2π .* 440.0f0 .* t))
    power_sine = NN.stft_power(sine)
    @test size(power_sine, 1) == 201
    @test maximum(power_sine) > 0
end

@testset "Log-Mel Spectrogram Shape and Range" begin
    # White noise
    rng_audio = Float32.(randn(NN.N_SAMPLES))
    mel = NN.log_mel_spectrogram(rng_audio)
    
    @test size(mel, 1) == 80
    @test eltype(mel) == Float32
    
    # After Whisper normalisation the output is typically in (-1, 1] range.
    # The max is not guaranteed to be ≤ 1.0 for arbitrary inputs,
    # but should be finite and in a reasonable range.
    @test maximum(mel) <= 2.0f0  # loose bound
    
    # Values should not be NaN or Inf
    @test !any(isnan.(mel))
    @test !any(isinf.(mel))
    
    # Test silence: all mel bins should be the minimum value
    silence = zeros(Float32, NN.N_SAMPLES)
    mel_silence = NN.log_mel_spectrogram(silence)
    @test all(mel_silence .== mel_silence[1, 1])
end

@testset "pad_or_trim" begin
    long_audio  = ones(Float32, NN.N_SAMPLES + 1000)
    short_audio = ones(Float32, 1000)
    exact_audio = ones(Float32, NN.N_SAMPLES)
    
    @test length(NN.pad_or_trim(long_audio))  == NN.N_SAMPLES
    @test length(NN.pad_or_trim(short_audio)) == NN.N_SAMPLES
    @test length(NN.pad_or_trim(exact_audio)) == NN.N_SAMPLES
    
    trimmed = NN.pad_or_trim(long_audio)
    @test all(trimmed .== 1.0f0)   # trimming keeps values
    
    padded = NN.pad_or_trim(short_audio)
    @test all(padded[1:1000] .== 1.0f0)  # leading values preserved
    @test all(padded[1001:end] .== 0.0f0) # trailing zeros added
end

@testset "Full pipeline: audio → log-mel → Luminal graph input" begin
    # Simulate 30s of audio, feed into the Luminal NN.AudioEncoder
    audio = Float32.(randn(NN.N_SAMPLES))
    mel   = NN.log_mel_spectrogram(audio)          # (80, 3000)
    
    # AudioEncoder expects (batch, mels, frames)
    input_data = Base.reshape(mel, 1, 80, size(mel, 2)) # (1, 80, 3000)
    
    @test size(input_data) == (1, 80, size(mel, 2))
    @test !any(isnan.(input_data))
    
    # Optionally: actually run through the Luminal AudioEncoder graph
    # (may be slow, so we keep it as a quick smoke-test)
    graph = Luminal.Graph()
    # Use small audio_frames to keep tests fast
    small_frames = 50
    small_input  = NN.log_mel_spectrogram(zeros(Float32, small_frames * NN.HOP_LENGTH))
    input_tensor = Luminal.tensor(graph, [1, 80, small_frames])
    
    encoder = NN.AudioEncoder(graph)
    enc_out = encoder(input_tensor)
    @test enc_out isa Luminal.GraphTensor
end
