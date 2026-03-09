#!/bin/bash

# List of scripts to run
SCRIPTS=(
    "hist-da.sh"
    "mnist-da-new.sh"
    "cifar10-da.sh"
    "gm.sh"
    "hist-rt.sh"
    "hist-rt-cpu.sh"
)

# Verify GPM_ENV conda environment is active
CURRENT_ENV=$(conda info --json | jq -r '.active_prefix_name')
if [[ "$CURRENT_ENV" != "GPM_ENV" ]]; then
    echo "Error: Please activate the GPM_ENV conda environment first." >&2
    exit 1
fi

# Run scripts with sbatch if available, otherwise run directly
if command -v sbatch &> /dev/null; then
    for script in "${SCRIPTS[@]}"; do
        sbatch "$script"
    done
else
    for script in "${SCRIPTS[@]}"; do
        ./"$script"
    done
fi
