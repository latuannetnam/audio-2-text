import argparse
from faster_whisper import WhisperModel

def transcribe_audio(audio_path, output_path, model_size="small"):
    # Initialize the model
    model = WhisperModel(model_size, device="cuda", compute_type="float32")
    
    # Transcribe the audio
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    print(f"Detected language '{info.language}' with probability {info.language_probability}")
    result = ""
    
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        result += segment.text + "\n"
    
    # Write the result to the specified output file
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(result)

def main():
     parser = argparse.ArgumentParser(description="Transcribe audio to text using Faster Whisper")
     parser.add_argument("-i", "--input", 
                        required=True,
                        help="Path to the input audio file")
     parser.add_argument("-o", "--output",
                         default="output.txt",
                        help="Path to the output text file")
     parser.add_argument("--model-size", 
                    choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
                    default="small",
                    help="Size of the Whisper model (default: small)")

     args = parser.parse_args()
    
     transcribe_audio(args.input, args.output, args.model_size)

if __name__ == "__main__":
    main()
