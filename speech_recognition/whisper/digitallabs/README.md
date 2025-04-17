# Project Setup Guide

## Local Development (Windows)

### Option 1: Using Conda (Recommended)

1. Create a new Conda environment with Python 3.11:
   ```bash
   conda create -n your_env_name python=3.11
   conda activate your_env_name
   ```

2. Install FFmpeg via Conda:
   ```bash
   conda install -c conda-forge ffmpeg
   ```

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Option 2: Manual Setup

1. Install FFmpeg:
   - Download FFmpeg from official website
   - Add FFmpeg to system PATH environment variables

2. Install Python 3.11

3. Setup virtual environment
   ```bash
   python -m venv your_venv_name
   .\your_venv_name\Scripts\activate
   ```

4. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

### Use CPU if you don't have a GPU

If you don't have a GPU, modify the following section in `requirements.txt`:

From:
```
--extra-index-url https://download.pytorch.org/whl/cu121
torch
torchvision
torchaudio
```

To:
```
torch
torchvision
torchaudio
```

This will install CPU-only versions of PyTorch packages.

## Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t streamlit-app .
   ```

2. Run the Docker container:
   ```bash
     docker run --gpus all -p 8501:8501 \
     -v $(pwd)/processed_data:/app/processed_data \
     -v $(pwd)/training:/app/training \
     -v $(pwd)/mounted_data:/app/mounted_data \
    --shm-size=16g \
     streamlit-app
   ```

   Note: Adjust `--shm-size` based on your model size:
   - Small models (<5GB): `--shm-size=8g`
   - Medium models (5-10GB): `--shm-size=16g`
   - Large models (>10GB): `--shm-size=32g` or higher

3. Access the application:
   ```
   http://localhost:8501
   ```
