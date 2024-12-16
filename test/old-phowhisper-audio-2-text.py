# Install required libraries
# !pip install transformers
# !pip install torch
# !pip install pydub
# !pip install librosa

from pydub import AudioSegment
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import librosa
import soundfile as sf
import torch


# Function to convert MP3 to WAV with 16kHz sample rate
def convert_mp3_to_wav(mp3_path, output_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
    audio.export(output_path, format="wav")

# Function to split audio into chunks
def split_audio(audio_path, chunk_length=30):
    audio, sr = librosa.load(audio_path, sr=16000)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    chunks = []
    
    for i in range(0, int(total_duration), chunk_length):
        start = i * sr
        end = min((i + chunk_length) * sr, len(audio))
        chunks.append(audio[start:end])
    
    return chunks, sr
    



# Get the uploaded MP3 file name
mp3_file = "vu-lan-muon-mang.mp3"

# Convert MP3 to WAV with 16 kHz sample rate
wav_file = "/content/audio_16kHz.wav"
convert_mp3_to_wav(mp3_file, wav_file)

# Load the Whisper model and processor
model_name = "vinai/PhoWhisper-small"

# Initialize the speech recognition pipeline with PhoWhisper-small
# transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small", device="cuda")
# output = transcriber(mp3_file)
# transcription = output['text']
# print(transcription)
# exit

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Split the audio into 30-second chunks
chunks, sr = split_audio(wav_file, chunk_length=30)

# Transcribe each chunk and combine results
transcription = ""
for i, chunk in enumerate(chunks):
    chunk_path = f"chunk_{i}.wav"
    sf.write(chunk_path, chunk, sr)  # Save the chunk as a file    
    # Preprocess the audio file
    audio_array, sr = librosa.load(chunk_path, sr=16000)
    input_features = processor(audio_array, sampling_rate=sr, return_tensors="pt").input_features

    # Generate transcription
    predicted_ids = model.generate(input_features)
    chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    transcription += chunk_transcription + " "
    print(f"{i + 1}: {chunk_transcription}")