#!/bin/bash
#SBATCH --time=2-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=236511M
#SBATCH --job-name=GPM
#SBATCH --output=logs/hist-da-%N-%j.out
#SBATCH --error=logs/hist-da-%N-%j.err

# Ensure current environment is GPM_ENV
CURRENT_ENV=$(conda info --json | jq -r '.active_prefix_name')
if [ "$CURRENT_ENV" != "GPM_ENV" ]; then
    echo "Please activate the GPM_ENV conda environment before running this script."
    exit 1
fi

# mkdir logs, if not exist
mkdir -p logs

# Delete the old logs/hist-da.csv, if exist
LOG_FILE=logs/hist-da.csv
if [ -f "$LOG_FILE" ]; then
    rm -f "$LOG_FILE"
fi

# Create a new logs/hist-da.log, witt certain content
echo "num_records,num_bins,beta,gamma,sigma,eps_comp,eps_actual,delta,success_rate,l2,gamma_hue" > $LOG_FILE

EXP_SCRIPT="hist-da.py"

NUM_RECORDS=1000000
REPEAT=1000
DELTA=1e-10

for num_bins in 256 4096 65536; do
    for eps in 0.125 0.25 0.5 1; do
        for beta in 1e-5 1e-4 1e-3 1e-2 1e-1; do
            # Run the python script with the parameters
            
            # get the sqrt of num_bins
            sqrt_num_bins=$(echo "scale=0; sqrt($num_bins)/1" | bc)
            for gamma_factor in 2 20; do
                gamma=$(echo "scale=10; $gamma_factor * $sqrt_num_bins" | bc)
                gamma_hue="${gamma_factor}\sqrt{d}"
                python $EXP_SCRIPT --num_records $NUM_RECORDS --num_bins $num_bins --epsilon $eps --delta $DELTA --beta $beta --gamma $gamma --repeat $REPEAT --gamma_hue "$gamma_hue" >> $LOG_FILE
                git add $LOG_FILE
                git commit -m "Add histogram results to $LOG_FILE, with num_bins=$num_bins, eps=$eps, beta=$beta, gamma=$gamma."
                git push
            done

            for gamma_factor in 2 20; do
                gamma=$(echo "scale=10; $gamma_factor * $num_bins" | bc)
                gamma_hue="${gamma_factor}d"
                python $EXP_SCRIPT --num_records $NUM_RECORDS --num_bins $num_bins --epsilon $eps --delta $DELTA --beta $beta --gamma $gamma --repeat $REPEAT --gamma_hue "$gamma_hue" >> $LOG_FILE
                git add $LOG_FILE
                git commit -m "Add histogram results to $LOG_FILE, with num_bins=$num_bins, eps=$eps, beta=$beta, gamma=$gamma."
                git push
            done
        done
    done
done

