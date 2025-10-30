#!/bin/bash

# SNLI Dataset Download and Setup Script

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$SCRIPT_DIR/../data"
DOWNLOAD_URL="https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
ZIP_FILE="$DATA_DIR/snli_1.0.zip"

echo "==================================="
echo "SNLI Dataset Download Script"
echo "==================================="

# Create data directory if it doesn't exist
echo "[1/4] Creating data directory..."
mkdir -p "$DATA_DIR"

# Download the dataset
echo "[2/4] Downloading SNLI dataset..."
if [ -f "$ZIP_FILE" ]; then
    echo "Dataset already downloaded. Skipping download."
else
    wget -O "$ZIP_FILE" "$DOWNLOAD_URL"
    echo "Download complete!"
fi

# Unzip the dataset
echo "[3/4] Extracting dataset..."
if [ -d "$DATA_DIR/snli_1.0" ]; then
    echo "Dataset already extracted. Skipping extraction."
else
    unzip -q "$ZIP_FILE" -d "$DATA_DIR"
    echo "Extraction complete!"
fi

# Clean up zip file (optional)
echo "[4/4] Cleaning up..."
read -p "Do you want to remove the zip file? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm "$ZIP_FILE"
    echo "Zip file removed."
else
    echo "Zip file kept at: $ZIP_FILE"
fi

echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo "SNLI dataset is now available at:"
echo "$DATA_DIR/snli_1.0/"
echo ""
echo "Dataset contents:"
ls -lh "$DATA_DIR/snli_1.0/" 2>/dev/null || echo "Directory listing not available"
echo ""
echo "You can now run the privacy-preserving experiments:"
echo "  bash DP/SNLI/scripts/central_private.sh"
echo "  bash DP/SNLI/scripts/fed_private.sh"
