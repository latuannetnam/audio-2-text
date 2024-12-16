from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy
import torch
import argparse
import os

def process_and_transcribe(audio_path, model, processor, device, chunk_length=30):
    """
    Process MP3 audio in chunks and transcribe each chunk
    audio_path: path to MP3 file
    model: loaded Whisper model
    processor: Whisper processor
    device: device to use for processing (cuda/cpu)
    chunk_length: length of each chunk in seconds
    """
    # Load the audio file
    audio = AudioSegment.from_mp3(audio_path)
    
    # Convert chunk_length from seconds to milliseconds
    chunk_length_ms = chunk_length * 1000
    
    transcription = ""
    chunk_count = len(audio) // chunk_length_ms + (1 if len(audio) % chunk_length_ms != 0 else 0)
    
    # Process each chunk
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        # Convert to 16kHz sample rate
        chunk = chunk.set_frame_rate(16000)
        
        # Convert to numpy array
        chunk_array = numpy.array(chunk.get_array_of_samples())
        
        # If stereo, convert to mono by averaging channels
        if chunk.channels == 2:
            chunk_array = chunk_array.reshape((-1, 2)).mean(axis=1)
        
        # Convert to float32 and normalize
        chunk_array = chunk_array.astype(numpy.float32) / 32768.0
        
        # Process chunk with Whisper
        input_features = processor(
            chunk_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features
        
        # Move input features to the same device as the model
        input_features = input_features.to(device)

        # Generate transcription for this chunk
        predicted_ids = model.generate(input_features)
        chunk_transcription = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        # Add chunk transcription to result
        transcription += chunk_transcription + " "
        
        # Print progress
        current_chunk = (i // chunk_length_ms) + 1
        print(f"Chunk {current_chunk}/{chunk_count}: {chunk_transcription}")
    
    return transcription.strip()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Transcribe MP3 audio file using PhoWhisper')
    parser.add_argument('-i','--input', required=True, help='Path to the input MP3 file')
    parser.add_argument('-o', '--output', help='Path to the output text file (default: input_filename.txt)')
    parser.add_argument('-c', '--chunk-length', type=int, default=30,
                        help='Length of audio chunks in seconds (default: 30)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    if not args.input.lower().endswith('.mp3'):
        print(f"Error: Input file '{args.input}' is not an MP3 file")
        return
    
    # Set default output file if not specified
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + '.txt'
    
    # Load the Whisper model and processor
    print("Loading model and processor...")
    model_name = "vinai/PhoWhisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    print(f"Processing audio file: {args.input}")
    # Process and transcribe the audio
    full_transcription = process_and_transcribe(args.input, model, processor, device, args.chunk_length)
    
    print("\nFull Transcription:")
    print(full_transcription)
    
    # Save the transcription to the output file
    print(f"\nSaving transcription to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(full_transcription)

if __name__ == "__main__":
    main()
