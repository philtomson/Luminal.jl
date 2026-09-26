# Transcribe audio files with a Hugging Face Whisper checkpoint.
#
#   git clone https://huggingface.co/openai/whisper-tiny
#   julia --project=. examples/whisper.jl whisper-tiny speech.wav [more.wav ...] [--cpu] [--translate]
#
# Audio is decoded and resampled to 16 kHz with ffmpeg; up to 30 s per file.
# Several files are transcribed as one batch.
using Luminal

model_dir = ARGS[1]
files = filter(a -> !startswith(a, "--"), ARGS[2:end])
device = "--cpu" in ARGS ? CPUDevice() : get_device()
task = "--translate" in ARGS ? :translate : :transcribe

session = WhisperSession(model_dir; device=device)
transcribe(session, files; task=task)                                   # compiles
t = @elapsed results = transcribe(session, files; task=task)
for (f, (text, tokens)) in zip(files, results)
    println(f, ": ", strip(text))
end
println("($(length(files)) file(s), $(round(t, digits=2)) s warm)")
