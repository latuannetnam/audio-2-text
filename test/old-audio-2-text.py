import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import os
import wave

def convert_audio_to_text(audio_file_path):
    # Check if Vietnamese model exists, if not download it
    model_path = "vosk-model-vn-0.4"
    if not os.path.exists(model_path):
        print("Vietnamese model not found. Please download it first from Vosk website.")
        print("Visit: https://alphacephei.com/vosk/models and download the Vietnamese model")
        return None
    
    # Load Vosk model
    model = Model(model_path)
    
    # Get file extension
    file_extension = os.path.splitext(audio_file_path)[1].lower()
    
    try:
        # Convert mp3/mp4 to wav format first
        if file_extension in ['.mp3', '.mp4']:
            print(f"Converting {file_extension} to WAV format...")
            audio = AudioSegment.from_file(audio_file_path)
            # Convert to mono and set sample rate to 16000
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            wav_path = "temp_audio.wav"
            audio.export(wav_path, format="wav")
            audio_file_path = wav_path

        # Process the audio file
        wf = wave.open(audio_file_path, "rb")
        if wf.getnchannels() != 1:
            print("Audio file must be mono")
            return None

        # Create recognizer
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        # Process audio file
        print("Converting speech to text...")
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if 'text' in result and result['text'].strip():
                    print(result['text'])
                    results.append(result['text'])

        # Get final result
        final_result = json.loads(rec.FinalResult())
        if 'text' in final_result and final_result['text'].strip():
            results.append(final_result['text'])

        return ' '.join(results)

    except Exception as e:
        return f"Error occurred: {e}"
    finally:
        # Clean up temporary WAV file if it was created
        # if 'wav_path' in locals() and os.path.exists(wav_path):
        #     os.remove(wav_path)
        print("Conversion completed.")

def save_text_to_file(text, output_file="output.txt"):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text has been saved to {output_file}")
    except Exception as e:
        print(f"Error saving text to file: {e}")

def main():
    # Get input file path from user
    audio_file_path = input("Enter the path to your audio file (MP3 or MP4): ")
    
    if not os.path.exists(audio_file_path):
        print("Error: File does not exist!")
        return
    
    # Convert audio to text
    print("Starting conversion...")
    text = convert_audio_to_text(audio_file_path)
    
    # Save the text to a file
    if text:
        print("\nConverted Text:")
        print("-" * 50)
        print(text)
        print("-" * 50)
        save_text_to_file(text)

if __name__ == "__main__":
    main()
