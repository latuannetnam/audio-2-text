import gradio as gr
import whisper
import threading
import time
import queue
from datetime import datetime
import os

# Global flag for stopping and message queue
stop_processing = False
processing_lock = threading.Lock()
message_queue = queue.Queue()

def transcribe_single_audio(audio_path, model, progress=gr.Progress(), total_files=1, current_file=1):
    global stop_processing
    
    # Get filename without path
    filename = os.path.basename(audio_path)
    
    # Transcribe the audio with segments
    progress(0.1, desc=f"[{current_file}/{total_files}] Starting transcription of {filename}...")
    result = model.transcribe(audio_path)
    
    # Initial language info
    detected_language = result["language"]
    language_message = f"\n=== File: {filename} ===\nDetected language: {detected_language}\n\n"
    message_queue.put(language_message)
    
    # Get total number of segments for progress calculation
    total_segments = len(result["segments"])
    
    # Process each segment
    for idx, segment in enumerate(result["segments"], 1):
        if stop_processing:
            message_queue.put("\n[Processing stopped by user]")
            return False
            
        # Calculate progress percentage for this file (from 10% to 100%)
        progress_value = 0.1 + (0.9 * idx / total_segments)
        progress(progress_value, desc=f"[{current_file}/{total_files}] Processing {filename} - segment {idx}/{total_segments}")
            
        # Format timestamp as [MM:SS.mmm]
        start_time = time.strftime('%M:%S', time.gmtime(segment['start']))
        end_time = time.strftime('%M:%S', time.gmtime(segment['end']))
        
        # Create segment text with timestamps
        segment_text = f"[{start_time} → {end_time}] {segment['text']}\n"
        segment_text = f"{segment['text']}\n"
        message_queue.put(segment_text)
        time.sleep(0.01)  # Small delay to make stop more responsive
    
    message_queue.put("\n")  # Add spacing between files
    return True

def transcribe_audio(audio_files, model_size="small", progress=gr.Progress()):
    global stop_processing
    
    if not audio_files:
        message_queue.put("No audio files provided.")
        return
    
    # Convert to list if single file
    if not isinstance(audio_files, list):
        audio_files = [audio_files]
    
    # Initialize the model
    progress(0, desc="Loading model...")
    model = whisper.load_model(model_size)    
    
    # Reset stop flag
    stop_processing = False    
    
    total_files = len(audio_files)
    
    # Process each audio file
    for idx, audio_file in enumerate(audio_files, 1):
        if stop_processing:
            break
            
        success = transcribe_single_audio(
            audio_file, 
            model, 
            progress, 
            total_files=total_files, 
            current_file=idx
        )
        
        if not success:
            break
    
    progress(1.0, desc="Transcription complete!")
    return

def stop_transcription():
    global stop_processing
    stop_processing = True
    return "Stopping transcription..."

def create_download(text):
    if not text or text.strip() == "":
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcript_{timestamp}.txt"
    
    # Create a temporary file with the content
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    
    return filename

def process_audio(audio_files, model_choice, progress=gr.Progress()):
    global stop_processing, message_queue
    
    if not audio_files:
        return "Please upload audio files.", "No files provided."
    
    # Check if already processing
    if not processing_lock.acquire(blocking=False):
        return "Processing already in progress. Please wait or stop current process.", "Busy"
    
    try:
        # Clear the message queue
        while not message_queue.empty():
            message_queue.get()
        
        # Start transcription in a separate thread
        thread = threading.Thread(
            target=transcribe_audio,
            args=(audio_files, model_choice, progress)
        )
        thread.start()
        
        # Collect and yield results
        full_text = ""
        while thread.is_alive() or not message_queue.empty():
            try:
                text = message_queue.get(timeout=0.1)
                full_text += text
                yield full_text, "Processing..."
            except queue.Empty:
                continue

        return full_text, "Complete"
            
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return error_message, "Error"
    finally:
        processing_lock.release()

# In the Gradio interface section:
with gr.Blocks() as app:
    gr.Markdown("# Audio Transcription using Whisper")
    gr.Markdown("Upload one or more audio files and select a model size to transcribe the audio to text.")
    
    with gr.Row():
        audio_input = gr.File(
            file_count="multiple",
            file_types=["audio"],
            label="Upload Audio Files"
        )
        model_choice = gr.Dropdown(
            choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
            value="turbo",
            label="Model Size"
        )
    
    with gr.Row():
        transcribe_btn = gr.Button("Transcribe")
        stop_btn = gr.Button("Stop")
    
    output_text = gr.Textbox(label="Transcription", lines=10)
    status_text = gr.Textbox(label="Status", lines=1)
    
    # Add download button
    download_btn = gr.Button("Download Transcription")
    download_output = gr.File(label="Download")
    
    # Set up button clicks
    transcribe_btn.click(
        fn=process_audio,
        inputs=[audio_input, model_choice],
        outputs=[output_text, status_text],
        show_progress=True
    ).then(
        fn=lambda: "Task done!",
        inputs=None,
        outputs=status_text
    )
    
    stop_btn.click(
        fn=stop_transcription,
        inputs=None,
        outputs=status_text
    )
    
    download_btn.click(
        fn=create_download,
        inputs=[output_text],
        outputs=download_output
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,       # Default Gradio port
        inbrowser=True         # Open in browser automatically
    )
