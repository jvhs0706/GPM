#!/bin/bash

# list of scripts to run
# ./hist-da.sh
# ./mnist-da-new.sh
# ./cifar10-da.sh
# ./gm.sh
# ./hist-rt.sh
# ./hist-rt-cpu.sh

# Ensure current environment is GPM_ENV
CURRENT_ENV=$(conda info --json | jq -r '.active_prefix_name')
if [ "$CURRENT_ENV" != "GPM_ENV" ]; then
    echo "Please activate the GPM_ENV conda environment before running this script."
    exit 1
fi

SCRIPTS=(
    "hist-da.sh"
    "mnist-da-new.sh"
    "cifar10-da.sh"
    "gm.sh"
    "hist-rt.sh"
    "hist-rt-cpu.sh"
)

# if slurm is available, run the scripts with sbatch, otherwise run them directly
if command -v sbatch &> /dev/null; then
    for script in "${SCRIPTS[@]}"; do
        sbatch $script
    done
else
    for script in "${SCRIPTS[@]}"; do
        # Run the script directly ./script
        ./$script
    done
fi