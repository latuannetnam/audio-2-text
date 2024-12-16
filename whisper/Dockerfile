# Use NVIDIA CUDA base image with cuDNN 9
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"
ENV FORCE_CUDA="1"

# Set working directory
WORKDIR /app

# Install system dependencies and Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    portaudio19-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user (letting the system assign UID)
RUN useradd -m appuser
RUN mkdir -p /app/models /app/transcripts && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy the application code
COPY --chown=appuser:appuser whisper-gradio-audio-2-text.py .

# Create volumes for models and transcripts
VOLUME ["/app/models", "/app/transcripts"]

# Expose the port Gradio will run on
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Command to run the application
CMD ["python3.12", "whisper-gradio-audio-2-text.py"]
