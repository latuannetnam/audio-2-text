# Convert audio file to text

This guide will help you set up the necessary environment and tools to convert audio files to text using OpenAI's Whisper model and PhoWhisper model.

## Preparation - Using WSL - Ubuntu 24.04

### Nvidia driver
First, install the Nvidia driver to enable GPU support.

```sh
yes | sudo apt install nvidia-driver-550
sudo reboot
```

### Install Cuda toolkit
Download and install the CUDA toolkit, which is required for GPU acceleration.

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

### Install Linux kernel header
Install the Linux kernel headers, which are necessary for building kernel modules.

```sh
yes | sudo apt install linux-headers-generic
```

### Set CUDA path
Add the CUDA binaries to your PATH and set the library path.

```sh
vim ~/.bashrc
```

Add the following lines to the end of the file:

```sh
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

Then, source the `.bashrc` file to apply the changes:

```sh
source ~/.bashrc
```

### Install Cuda container toolkit
Install the CUDA container toolkit to enable GPU support in Docker containers.

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### FFMPEG
Install FFMPEG, a tool for handling multimedia data.

```sh
sudo apt install ffmpeg
```

### Python virtual environment
Create and activate a Python virtual environment to manage dependencies.

```sh
python -m venv venv
source ~/venv/bin/activate
```

## Use OpenAI Whisper model
Navigate to the Whisper directory and install the required Python packages.

```sh
cd whisper
pip install -r requirements.txt
```

### Using Whisper command
Run the Whisper model using the provided script.

```sh
./whisper-run.sh
```

### Using Gradio
Run the Gradio interface for the Whisper model.

```sh
./gradio-whisper.sh
```

## Use PhoWhisper model
Navigate to the PhoWhisper directory and install the required Python packages.

```sh
cd PhoWhisper
pip install -r requirements.txt
```
### PhoWhisper command line arguments

The `phowhisper-audio-2-text.py` script accepts several command line arguments to customize its behavior. Here are the available options:

- `-i`, `--input`: Specify the input audio file. This argument is required.
- `-o`, `--output`: Specify the output text file. If not provided, the output will be printed to the console.
- `-m`, `--model`: Specify the model to use. Default is `tiny`.
Example usage:

```sh
python phowhisper-audio-2-text.py -i audio.mp3 -o transcript.txt -m large
```

## Trobleshoot

### Error
Intel oneMKL ERROR: Parameter 3 was incorrect on entry to SGEBAL

Run command to force install torch

```sh
pip install --force-reinstall torch torchvision torchaudio
```

Or re-run the script again