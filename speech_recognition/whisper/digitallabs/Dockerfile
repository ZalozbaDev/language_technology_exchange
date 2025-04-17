# Use an official Python 3.11 slim image as the base image
FROM python:3.11-slim

# Set environment variables to reduce output noise during installation
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install FFmpeg and other necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv or other Python dependency management tools if needed
RUN pip install --no-cache-dir --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Create necessary directories
RUN mkdir -p processed_data training/models uploaded_data

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Set permissions for the directories
RUN chmod -R 777 processed_data training uploaded_data

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--browser.gatherUsageStats", "false"]