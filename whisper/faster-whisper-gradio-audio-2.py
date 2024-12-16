import gradio as gr
from faster_whisper import WhisperModel
import threading
import time
import queue
from datetime import datetime

# Global flag for stopping and message queue
stop_processing = False
processing_lock = threading.Lock()
message_queue = queue.Queue()

def transcribe_audio(audio_path, model_size="small", progress=gr.Progress()):
    global stop_processing
    
    # Initialize the model
    model = WhisperModel(model_size, device="cuda", compute_type="float32")
    
    # Reset stop flag
    stop_processing = False
    
    # Transcribe the audio
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    # Initial language info
    result = f"Detected language: {info.language} (probability: {info.language_probability:.2f})\n\n"
    message_queue.put(result)
    
    # Process each segment
    for segment in segments:
        if stop_processing:
            message_queue.put("\n[Processing stopped by user]")
            break
            
        message_queue.put(f"{segment.text} ")
        time.sleep(0.01)  # Small delay to make stop more responsive
    
    return

def process_audio(audio_file, model_choice):
    global stop_processing, message_queue
    
    if audio_file is None:
        return "Please upload an audio file."
    
    # Check if already processing
    if not processing_lock.acquire(blocking=False):
        return "Processing already in progress. Please wait or stop current process."
    
    try:
        # Clear the message queue
        while not message_queue.empty():
            message_queue.get()
        
        # Start transcription in a separate thread
        thread = threading.Thread(
            target=transcribe_audio,
            args=(audio_file, model_choice)
        )
        thread.start()
        
        # Collect and yield results
        full_text = ""
        while thread.is_alive() or not message_queue.empty():
            try:
                text = message_queue.get(timeout=0.1)
                full_text += text
                yield full_text
            except queue.Empty:
                continue
            
        return full_text
            
    except Exception as e:
        return f"An error occurred: {str(e)}"
    finally:
        processing_lock.release()

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

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Audio Transcription using Faster Whisper")
    gr.Markdown("Upload an audio file and select a model size to transcribe the audio to text.")
    
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
        model_choice = gr.Dropdown(
            choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
            value="small",
            label="Model Size"
        )
    
    with gr.Row():
        transcribe_btn = gr.Button("Transcribe")
        stop_btn = gr.Button("Stop")
    
    output_text = gr.Textbox(label="Transcription", lines=10)
    
    # Add download button
    download_btn = gr.Button("Download Transcription")
    download_output = gr.File(label="Download")
    
    # Set up button clicks
    transcribe_btn.click(
        fn=process_audio,
        inputs=[audio_input, model_choice],
        outputs=output_text,
        show_progress=True
    )
    
    stop_btn.click(
        fn=stop_transcription,
        inputs=None,
        outputs=output_text
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
        # share=True,          # Get a public URL
        inbrowser=True         # Open in browser automatically
    )
