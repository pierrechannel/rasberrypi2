FROM arm64v8/python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    cmake \
    libjpeg-dev \
    libpng-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip
# Install numpy first to avoid build issues
RUN pip install --no-cache-dir numpy==1.26.4
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .