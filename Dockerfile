# Start from a slim Python base image, specifying the required AMD64 platform
FROM --platform=linux/amd64 python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app
COPY wheelhouse/ ./wheelhouse
# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the Python dependencies. This happens offline during the build.
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-index --find-links=./wheelhouse -r requirements.txt

# Copy your trained model and the inference script into the container
# The '.' copies everything from the current directory on your machine to /app in the container
COPY ./inference.py /app/inference.py
COPY ./models /app/models

# Set the default command to run when the container starts.
# This script must be written to read from /app/input and write to /app/output.
CMD ["python", "inference.py"]