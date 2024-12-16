from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import snapshot_download, try_to_load_from_cache
import numpy
import torch
import argparse
import os
import sys
from gec_model import GecBERTModel
def ensure_model_downloaded(model_name):
    """
    Check if model is cached and download if not present
    """
    try:
        print(f"Checking/downloading model {model_name}...")
        snapshot_download(
            repo_id=model_name,
            local_dir=None,  # Will use default cache directory
            local_dir_use_symlinks=False
        )
        print(f"Model {model_name} ready!")

    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)


def load_models():
    """Load both Whisper and punctuation models"""
    try:
        # Load Whisper model and processor
        print("Loading Whisper model and processor...")
        # whisper_model_name = "vinai/PhoWhisper-small"
        whisper_model_name = "vinai/PhoWhisper-medium"
        processor = WhisperProcessor.from_pretrained(whisper_model_name)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        
        # Load punctuation model
        print("Loading punctuation model...")
        punct_model_name = "dragonSwing/xlm-roberta-capu"
        ensure_model_downloaded(punct_model_name)

        try:
            punct_model = GecBERTModel(
                vocab_path="vocabulary",
                model_paths=punct_model_name,
                split_chunk=True
            )
        except Exception as e:
            print(f"Error loading punctuation model: {e}")
            sys.exit(1)

        # Move models to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        whisper_model = whisper_model.to(device)
        punct_model = punct_model.to(device)

        return whisper_model, punct_model, processor, device

    except Exception as e:
        print(f"Error in load_models: {e}")
        sys.exit(1)

def restore_punctuation(text, punct_model, device):
    """
    Restore punctuation in the given text using the punctuation model
    """
    result = punct_model(text)
    # If result is a list, join it into a single string
    if isinstance(result, list):
        result = ' '.join(result)
    return result


def process_and_transcribe(audio_path, model, punct_model, processor, device, chunk_length=30):
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
    
    # Load all models
    print("Loading models...")
    model, punct_model, processor, device = load_models()
    
    print(f"Processing audio file: {args.input}")
    # Process and transcribe the audio
    full_transcription = process_and_transcribe(args.input, model, punct_model, processor, device, args.chunk_length)
    full_transcription = restore_punctuation(full_transcription, punct_model, device)
    
    print("\nFull Transcription:")
    print(full_transcription)
    
    # Save the transcription to the output file
    print(f"\nSaving transcription to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(full_transcription)

if __name__ == "__main__":
    main()
