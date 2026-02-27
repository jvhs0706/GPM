#!/bin/bash
#SBATCH --time=2-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=GPM
#SBATCH --output=logs/plot-%N-%j.out
#SBATCH --error=logs/plot-%N-%j.err

Ensure current environment is GPM_ENV
CURRENT_ENV=$(conda info --json | jq -r '.active_prefix_name')
if [ "$CURRENT_ENV" != "GPM_ENV" ]; then
    echo "Please activate the GPM_ENV conda environment before running this script."
    exit 1
fi

python plot-hclwe.py
python plot-intuition.py
python plot-privacy-params.py
python plot-hist-rt-new.py hist-rt hist-rt-cpu
python plot-hist-new.py hist-da
python plot-mnist-new.py mnist-da-beta-by-2


git add plots/*.png
git add plots/*.pdf
git commit -m "Add plots"
git push
