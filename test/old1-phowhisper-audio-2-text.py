from pydub import AudioSegment
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    XLMRobertaTokenizer,
    XLMRobertaForTokenClassification
)
from huggingface_hub import snapshot_download, try_to_load_from_cache
import numpy
import torch
import argparse
import os
import sys
from gec_model import GecBERTModel

def initialize_cuda():
    """Initialize and verify CUDA setup"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        return "cpu"
    
    try:
        # Test CUDA initialization
        torch.cuda.init()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Using CUDA device: {device_name}")
        return "cuda"
    except Exception as e:
        print(f"CUDA initialization error: {e}")
        print("Falling back to CPU")
        return "cpu"


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
        # Load Whisper model
        print("Loading Whisper model and processor...")
        whisper_model_name = "vinai/PhoWhisper-small"
        ensure_model_downloaded(whisper_model_name)

        try:
            whisper_processor = WhisperProcessor.from_pretrained(
                whisper_model_name)
            whisper_model = WhisperForConditionalGeneration.from_pretrained(
                whisper_model_name)
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            sys.exit(1)

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

        # Initialize CUDA
        device = initialize_cuda()
        
        # Move models to device
        whisper_model = whisper_model.to(device)
        punct_model = punct_model.to(device)

        # Only convert to half precision if using CUDA
        if device == "cuda":
            try:
                whisper_model = whisper_model.half()
                punct_model = punct_model.half()
            except Exception as e:
                print(f"Warning: Could not convert to half precision: {e}")

        return whisper_processor, whisper_model, punct_model, device

    except Exception as e:
        print(f"Error in load_models: {e}")
        sys.exit(1)


def restore_punctuation(text, punct_model, device):
    """
    Restore punctuation in the given text using the punctuation model
    """
    result = punct_model(text)

    return result


def process_and_transcribe(audio_path, whisper_model, whisper_processor, punct_model, punct_tokenizer, device, chunk_length=30):
    """
    Process MP3 audio in chunks, transcribe each chunk, and add punctuation
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_mp3(audio_path)

        # Convert chunk_length from seconds to milliseconds
        chunk_length_ms = chunk_length * 1000

        transcription = ""
        chunk_count = len(audio) // chunk_length_ms + \
            (1 if len(audio) % chunk_length_ms != 0 else 0)

        # Process each chunk
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunk = chunk.set_frame_rate(16000)

            # Convert to numpy array
            chunk_array = numpy.array(chunk.get_array_of_samples())

            if chunk.channels == 2:
                chunk_array = chunk_array.reshape((-1, 2)).mean(axis=1)

            chunk_array = chunk_array.astype(numpy.float32) / 32768.0

            # Process chunk with Whisper
            input_features = whisper_processor(
                chunk_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features

            input_features = input_features.to(device)

            # Generate transcription for this chunk
            predicted_ids = whisper_model.generate(input_features)
            chunk_transcription = whisper_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            # Add punctuation to the chunk
            chunk_transcription = restore_punctuation(
                chunk_transcription,
                punct_model,
                device
            )

            transcription += chunk_transcription + " "

            # Print progress
            current_chunk = (i // chunk_length_ms) + 1
            print(
                f"Chunk {current_chunk}/{chunk_count}: {chunk_transcription}")

        return transcription.strip()

    except Exception as e:
        print(f"Error processing audio: {e}")
        sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Transcribe MP3 audio file using PhoWhisper with punctuation restoration')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the input MP3 file')
    parser.add_argument(
        '-o', '--output', help='Path to the output text file (default: input_filename.txt)')
    parser.add_argument('-c', '--chunk-length', type=int, default=30,
                        help='Length of audio chunks in seconds (default: 30)')
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return

    if not args.input.lower().endswith('.mp3'):
        print(f"Error: Input file '{args.input}' is not an MP3 file")
        return

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + '.txt'

    # Load all models
    whisper_processor, whisper_model,  punct_model, device = load_models()

    print(f"Processing audio file: {args.input}")
    # Process and transcribe the audio with punctuation
    full_transcription = process_and_transcribe(
        args.input,
        whisper_model,
        whisper_processor,
        punct_model,
        device,
        args.chunk_length
    )

    print("\nFull Transcription:")
    print(full_transcription)

    # Save the transcription to the output file
    print(f"\nSaving transcription to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(full_transcription)


if __name__ == "__main__":
    main()
