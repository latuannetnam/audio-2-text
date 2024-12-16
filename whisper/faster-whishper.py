from faster_whisper import WhisperModel

# model_size = "large-v3"
# model_size = "turbo"
model_size = "small"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8")
model = WhisperModel(model_size, device="cuda", compute_type="float32")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Get the audio file path from the user
audio_file_path = input("Enter the path to your audio file: ")
segments, info = model.transcribe(audio_file_path, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
result =""

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    result += segment.text + "\n"

# Write the result to a text file
with open("output.txt", "w") as f:
    f.write(result)
    