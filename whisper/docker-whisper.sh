#docker build -t whisper-transcription-gpu .

docker run -d \
  --name whisper-transcription-gpu \
  --gpus all \
  -p 8001:7860 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/transcripts:/app/transcripts \
  --restart unless-stopped \
  whisper-transcription-gpu:latest
