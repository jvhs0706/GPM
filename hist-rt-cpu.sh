#!/bin/bash
#SBATCH --time=2-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=236511M
#SBATCH --job-name=GPM
#SBATCH --output=logs/hist-rt-%N-%j.out
#SBATCH --error=logs/hist-rt-%N-%j.err

# Ensure current environment is GPM_ENV
CURRENT_ENV=$(conda info --json | jq -r '.active_prefix_name')
if [ "$CURRENT_ENV" != "GPM_ENV" ]; then
    echo "Please activate the GPM_ENV conda environment before running this script."
    exit 1
fi

# mkdir logs, if not exist
mkdir -p logs

# Delete the old logs/hist-da.csv, if exist
LOG_FILE=logs/hist-rt-cpu.csv
if [ -f "$LOG_FILE" ]; then
    rm -f "$LOG_FILE"
fi

# Create a new logs/hist-da.log, witt certain content
echo "num_records,num_bins,beta,gamma,sigma,eps_comp,eps_actual,delta,mech,rt" > $LOG_FILE

EXP_SCRIPT="hist-rt.py"
REPEAT=100
DELTA=1e-10
NUM_RECORDS=114514

for num_bins in 65536 4096 256; do
    for eps in 0.125 0.25 0.5 1; do
        for beta in 1e-5 1e-4 1e-3 1e-2 1e-1; do
            # Run the python script with the parameters
            python $EXP_SCRIPT --num_records $NUM_RECORDS --num_bins $num_bins --epsilon $eps --delta $DELTA --beta $beta --repeat $REPEAT --device cpu >> $LOG_FILE
            python $EXP_SCRIPT --num_records $NUM_RECORDS --num_bins $num_bins --epsilon $eps --delta $DELTA --beta $beta --repeat $REPEAT --gpm --device cpu >> $LOG_FILE
            git add $LOG_FILE
            git commit -m "Add histogram noise sampling time results to $LOG_FILE, with $num_bins bins, eps=$eps, beta=$beta."
            git push
        done
    done
done

