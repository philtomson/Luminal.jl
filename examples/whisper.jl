# Transcribe an audio file with a Hugging Face Whisper checkpoint.
#
#   git clone https://huggingface.co/openai/whisper-tiny
#   julia --project=. examples/whisper.jl whisper-tiny speech.wav [cpu|gpu] [--translate]
#
# Audio is decoded and resampled to 16 kHz with ffmpeg; up to 30 s is transcribed.
using Luminal

model_dir, audio_path = ARGS[1], ARGS[2]
device = "cpu" in ARGS ? CPUDevice() : get_device()
task = "--translate" in ARGS ? :translate : :transcribe

audio = Luminal.NN.load_audio_file(audio_path)
text, tokens = transcribe(model_dir, audio; device=device, task=task)   # compiles
t = @elapsed text, tokens = transcribe(model_dir, audio; device=device, task=task)
println(strip(text))
println("($(length(tokens)) tokens, $(round(t, digits=2)) s warm)")
