#!/bin/bash
#SBATCH --time=2-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=236511M
#SBATCH --job-name=GPM
#SBATCH --output=logs/mnist-da-%N-%j.out
#SBATCH --error=logs/mnist-da-%N-%j.err

# Ensure current environment is GPM_ENV
CURRENT_ENV=$(conda info --json | jq -r '.active_prefix_name')
if [ "$CURRENT_ENV" != "GPM_ENV" ]; then
    echo "Please activate the GPM_ENV conda environment before running this script."
    exit 1
fi

# mkdir logs, if not exist
mkdir -p logs

# Delete the old logs/mnist-da.csv, if exist
LOG_FILE=logs/mnist-da-beta-by-2.csv
if [ -f "$LOG_FILE" ]; then
    rm -f "$LOG_FILE"
fi

# Create a new logs/mnist-da.log, witt certain content
echo "dataset,model,num_params,converged,acc,batch_size,clip_norm,beta,gamma,sigma,eps_comp,eps_actual,delta,success_rate" > $LOG_FILE

EXP_SCRIPT="mnist-da.py"

REPEAT=100
DELTA=1e-10
BATCH_SIZE=128

for eps in 0.125 0.25 0.5 1; do
    for exp_beta in $(seq 17 -1 10); do
        beta=$(echo "scale=20; 2^(-$exp_beta)" | bc -l)
        for clip_norm in 4 8; do
            for gamma_factor in 2 20 200; do
                python $EXP_SCRIPT --batch_size $BATCH_SIZE --epsilon $eps --delta $DELTA --beta $beta --gamma_factor $gamma_factor --clip_norm $clip_norm --pretrained --repeat $REPEAT >> $LOG_FILE
                python $EXP_SCRIPT --batch_size $BATCH_SIZE --epsilon $eps --delta $DELTA --beta $beta --gamma_factor $gamma_factor --clip_norm $clip_norm --repeat $REPEAT >> $LOG_FILE
                git add $LOG_FILE
                git commit -m "Add mnist results to $LOG_FILE, with eps=$eps, beta=$beta, clip_norm=$clip_norm, gamma_factor=$gamma_factor"
                git push
            done
        done
    done
done

git add $LOG_FILE
git commit -m "Add mnist results to $LOG_FILE"
git push