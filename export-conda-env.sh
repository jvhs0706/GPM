#!/bin/bash
# This script exports the current conda environment to a YAML file.

# Check if conda is installed, and make sure the current environment is GPM_ENV

if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Conda and try again."
    exit 1
fi

CURRENT_ENV=$(conda info --json | jq -r '.active_prefix_name')
if [ "$CURRENT_ENV" != "GPM_ENV" ]; then
    echo "Please activate the GPM_ENV conda environment before running this script."
    exit 1
fi

# Export the current conda environment to a env.yaml, excluding the prefix line
conda env export --no-builds | grep -v "prefix: " > env.yaml
echo "Conda environment exported to env.yaml"

# Append a new line "      - --index-url https://download.pytorch.org/whl/cu124" to the end of the file
echo "      - --index-url https://download.pytorch.org/whl/cu124" >> env.yaml
echo "Added PyTorch index URL to env.yaml"