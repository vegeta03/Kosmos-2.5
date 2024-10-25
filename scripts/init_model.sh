#!/bin/bash

# Check if model exists
if [ ! -d "./model" ] || [ -z "$(ls -A ./model)" ]; then
    echo "Model directory is empty. Downloading model..."
    python download_model.py
else
    echo "Model already exists in mounted volume"
fi